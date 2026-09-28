import json
import os
import smtplib
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

SENT_HISTORY_FILE = "sent_history.json"
TARGET_ARTICLE_COUNT = 10
PUBMED_SEARCH_TERM = (
    '("bone health" OR "calcium" OR "bioavailability" OR "osteoporosis") '
    'AND "journal article"[PT]'
)

# ---------------------------------------------------------------------------
# DEDUPLICATION HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def load_sent_pmids(filepath: str) -> set:
    """Load previously processed PMIDs from local JSON store."""
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set()
    except Exception as err:
        print(f"[WARNING] Could not read {filepath}: {err}. Starting with empty set.")
        return set()


def save_sent_pmids(filepath: str, sent_pmids: set) -> None:
    """Persist updated set of processed PMIDs to local JSON store."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sorted(list(sent_pmids)), f, indent=2)
        print(f"[INFO] Successfully updated {filepath}.")
    except Exception as err:
        print(f"[ERROR] Failed to update {filepath}: {err}")

# ---------------------------------------------------------------------------
# PUBMED INGESTION (E-UTILITIES)
# ---------------------------------------------------------------------------
def fetch_pubmed_articles(history_pmids: set, target_count: int = 10) -> list:
    """
    Fetch unique, unsent PubMed articles from the last 10 years.
    Returns a list of dicts: [{'pmid': ..., 'title': ..., 'abstract': ..., 'journal': ..., 'pub_date': ..., 'url': ...}]
    """
    current_year = datetime.now().year
    min_year = current_year - 10

    # Step 1: ESearch to get matching PMIDs sorted by publication date
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": PUBMED_SEARCH_TERM,
        "mindate": f"{min_year}/01/01",
        "maxdate": f"{current_year}/12/31",
        "datetype": "pdat",
        "retmode": "json",
        "retmax": 200,  # Request enough candidates to allow deduplication
        "sort": "pub_date",
    }

    print(f"[INFO] Searching PubMed for articles from {min_year} to {current_year}...")
    resp = requests.get(esearch_url, params=esearch_params, timeout=30)
    resp.raise_for_status()
    search_data = resp.json()

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    print(f"[INFO] Retrieved {len(id_list)} candidate PMIDs from search.")

    # Filter out already sent PMIDs
    candidate_pmids = [pmid for pmid in id_list if pmid not in history_pmids]
    selected_pmids = candidate_pmids[:target_count]

    if not selected_pmids:
        print("[INFO] No new unsent articles found matching criteria.")
        return []

    print(f"[INFO] Fetching details for {len(selected_pmids)} PMIDs...")

    # Step 2: EFetch to get XML metadata & abstracts for selected PMIDs
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "id": ",".join(selected_pmids),
        "retmode": "xml",
    }

    efetch_resp = requests.get(efetch_url, params=efetch_params, timeout=30)
    efetch_resp.raise_for_status()

    # Parse XML response
    root = ET.fromstring(efetch_resp.text)
    articles = []

    for article_node in root.findall(".//PubmedArticle"):
        pmid_elem = article_node.find(".//MedlineCitation/PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        title_elem = article_node.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "No Title Available"

        # Construct full abstract text if broken into multiple sections
        abstract_nodes = article_node.findall(".//AbstractText")
        abstract_parts = [node.text for node in abstract_nodes if node.text]
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."

        journal_elem = article_node.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

        # Extract publication year
        pub_year_elem = article_node.find(".//JournalIssue/PubDate/Year")
        if pub_year_elem is not None and pub_year_elem.text:
            pub_date = pub_year_elem.text
        else:
            pub_date = str(current_year)

        articles.append({
            "pmid": pmid,
            "title": title.rstrip("."),
            "abstract": abstract,
            "journal": journal,
            "pub_date": pub_date,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return articles

# ---------------------------------------------------------------------------
# GEMINI AI SUMMARIZATION
# ---------------------------------------------------------------------------
def generate_digest_summary(articles: list) -> str:
    """
    Generate a 2-to-3 sentence executive overview from fetched articles using Google GenAI SDK.
    Includes automated fallback handling across primary and fallback endpoints.
    """
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY missing. Returning fallback overview.")
        return (
            "Today's digest presents recent clinical literature evaluating bone mineral density, "
            "calcium supplementation formulations, and therapeutic protocols for osteoporosis. "
            "Key studies emphasize optimizing bioavailability to improve patient outcomes in metabolic bone health."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Context compilation
    corpus = []
    for idx, art in enumerate(articles, start=1):
        corpus.append(f"--- Article {idx} ---")
        corpus.append(f"Title: {art['title']}")
        corpus.append(f"Abstract: {art['abstract']}\n")

    combined_text = "\n".join(corpus)

    prompt = (
        "You are an expert medical editor specializing in endocrinology and bone metabolism. "
        "Read the following recent article abstracts and write a clear, concise executive summary "
        "of exactly 2 to 3 sentences synthesis for a healthcare provider audience. Focus strictly "
        "on advancements in bone health, calcium formulation bioavailability, and osteoporosis management. "
        "Do not use markdown formatting, bullet points, or intros/outros.\n\n"
        f"Articles Content:\n{combined_text}"
    )

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    for model_name in models_to_try:
        try:
            print(f"[INFO] Generating AI summary using model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=250,
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception as err:
            print(f"[WARNING] Model {model_name} execution failed: {err}")

    # Default fallback if all API calls fail
    return (
        "Today's medical digest highlights advances in calcium bioavailability, bone tissue regeneration, "
        "and therapeutic interventions for clinical osteoporosis management. The selected literature provides "
        "critical insights into optimizing dietary and pharmaceutical formulations to support systemic bone density."
    )

# ---------------------------------------------------------------------------
# HTML BODY TEMPLATING
# ---------------------------------------------------------------------------
def build_email_html(summary: str, articles: list) -> str:
    """Construct a clean, responsive HTML email body."""
    date_str = datetime.now().strftime("%B %d, %Y")

    items_html = ""
    for idx, art in enumerate(articles, start=1):
        items_html += f"""
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 16px 0;">
                <div style="font-size: 11px; font-weight: 700; color: #0284c7; text-transform: uppercase; letter-spacing: 0.5px;">
                    Article #{idx} &bull; {art['journal']} ({art['pub_date']})
                </div>
                <h3 style="margin: 6px 0 8px 0; font-size: 16px; color: #1e293b; line-height: 1.4;">
                    <a href="{art['url']}" style="color: #0f172a; text-decoration: none;" target="_blank">
                        {art['title']}
                    </a>
                </h3>
                <p style="margin: 0 0 10px 0; font-size: 13px; color: #475569; line-height: 1.5;">
                    {art['abstract'][:300]}...
                </p>
                <div>
                    <a href="{art['url']}" style="font-size: 12px; color: #2563eb; text-decoration: none; font-weight: 600;" target="_blank">
                        Read Full Article on PubMed &rarr;
                    </a>
                </div>
            </td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Medical Research Digest</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f8fafc; padding: 24px 0;">
        <tr>
            <td align="center">
                <table role="presentation" width="100%" style="max-width: 640px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;" cellspacing="0" cellpadding="0">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background-color: #0f172a; padding: 28px 32px; text-align: left;">
                            <div style="font-size: 12px; font-weight: 600; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px;">
                                Daily Clinical Literature Digest
                            </div>
                            <h1 style="margin: 6px 0 0 0; font-size: 22px; color: #ffffff; font-weight: 700;">
                                Bone Health & Calcium Metabolism
                            </h1>
                            <div style="margin-top: 4px; font-size: 13px; color: #94a3b8;">
                                {date_str}
                            </div>
                        </td>
                    </tr>

                    <!-- Executive Overview -->
                    <tr>
                        <td style="padding: 24px 32px; background-color: #f0f9ff; border-bottom: 1px solid #e0f2fe;">
                            <div style="font-size: 11px; font-weight: 700; color: #0369a1; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                                Executive Synthesis
                            </div>
                            <p style="margin: 0; font-size: 14px; color: #0c4a6e; line-height: 1.6; font-style: italic;">
                                "{summary}"
                            </p>
                        </td>
                    </tr>

                    <!-- Main Content / Articles -->
                    <tr>
                        <td style="padding: 16px 32px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                                {items_html}
                            </table>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 20px 32px; background-color: #f1f5f9; text-align: center; font-size: 12px; color: #64748b; border-top: 1px solid #e2e8f0;">
                            Automated pipeline powered by PubMed E-Utilities & Google GenAI.<br>
                            To manage subscriptions or parameters, update your repository settings.
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    return html

# ---------------------------------------------------------------------------
# SMTP DISPATCH
# ---------------------------------------------------------------------------
def send_email(subject: str, html_body: str) -> None:
    """Send HTML email via Gmail SMTP using SSL (port 465)."""
    if not all([SENDER_EMAIL, GMAIL_APP_PASSWORD, RECEIVER_EMAIL]):
        raise ValueError("Missing essential SMTP credentials in environment variables.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    # Attach HTML payload
    msg.attach(MIMEText(html_body, "html"))

    print(f"[INFO] Connecting to smtp.gmail.com:465 (SSL) to send email...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())

    print(f"[SUCCESS] Digest email successfully dispatched to {RECEIVER_EMAIL}.")

# ---------------------------------------------------------------------------
# MAIN EXECUTION ORCHESTRATION
# ---------------------------------------------------------------------------
def main():
    print("=== STARTING MEDICAL RESEARCH DIGEST PIPELINE ===")

    # 1. Load sent history
    sent_pmids = load_sent_pmids(SENT_HISTORY_FILE)
    print(f"[INFO] Loaded {len(sent_pmids)} previously processed PMIDs.")

    # 2. Ingest articles from PubMed
    articles = fetch_pubmed_articles(sent_pmids, target_count=TARGET_ARTICLE_COUNT)

    if not articles:
        print("[INFO] No new articles fetched. Terminating execution without sending email.")
        sys.exit(0)

    # 3. Generate AI Executive Summary
    summary = generate_digest_summary(articles)

    # 4. Render HTML Body
    date_formatted = datetime.now().strftime("%Y-%m-%d")
    email_subject = f"Medical Research Digest: Bone Health & Bioavailability ({date_formatted})"
    html_body = build_email_html(summary, articles)

    # 5. Dispatch Email
    send_email(email_subject, html_body)

    # 6. Record sent PMIDs and update history file
    new_pmids = {art["pmid"] for art in articles if art.get("pmid")}
    updated_pmids = sent_pmids.union(new_pmids)
    save_sent_pmids(SENT_HISTORY_FILE, updated_pmids)

    print("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
