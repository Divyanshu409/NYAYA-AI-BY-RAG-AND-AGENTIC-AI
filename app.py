from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
import os
print("Gemini key found:", bool(os.getenv("GEMINI_API_KEY")))
print("Model:", os.getenv("GEMINI_MODEL"))

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict  
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import rag_engine

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "Nyaya AI is an educational civic-awareness assistant, not a lawyer, court, "
    "police authority, or government authority. This output is informational only "
    "and should not be treated as legal advice or a finding of guilt/liability."
)

LOW_CONFIDENCE_THRESHOLD = 45
HIGH_CONFIDENCE_THRESHOLD = 70
MAX_INPUT_CHARS = 5000

# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    language_hint: Optional[str] = Field(
        None,
        description="Optional: en, hi, ta, te, bn, mr, gu, pa, kn, ml, hinglish",
    )


class EntityExtraction(BaseModel):
    money_amounts: List[str] = []
    dates_or_time_refs: List[str] = []
    locations: List[str] = []
    organizations: List[str] = []
    contact_indicators_present: bool = False
    evidence_indicators: List[str] = []
    sensitive_terms: List[str] = []


class MatchedDomain(BaseModel):
    id: str
    label: str
    score: float
    explanation: str
    matched_signals: List[str]


class RightsMapping(BaseModel):
    title: str
    reference: str
    relevance: str
    confidence: str


class AnalysisResponse(BaseModel):
    disclaimer: str
    normalized_summary: str
    detected_language: str
    primary_category: str
    domains: List[MatchedDomain]
    possible_rights_or_law_domains: List[RightsMapping]
    risk_level: RiskLevel
    risk_reasons: List[str]
    confidence_score: int
    uncertainty_mode: bool
    uncertainty_note: Optional[str]
    extracted_context: EntityExtraction
    safety_notes: List[str]
    clarifying_questions: List[str]


# ---------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) schema
# Extends AnalysisResponse with an AI-written explanation + the knowledge-base
# sources it was grounded in. See rag_engine.py.
# ---------------------------------------------------------------------------


class RagSource(BaseModel):
    title: str
    snippet: str


class SelfCheckResult(BaseModel):
    ok: Optional[bool] = None          # None = not run (no Gemini key)
    issues: List[str] = []
    revised: bool = False              # True if the explanation was corrected


class RagAnalysisResponse(AnalysisResponse):
    ai_available: bool
    ai_explanation: str
    ai_sources: List[RagSource]
    self_check: SelfCheckResult = Field(default_factory=SelfCheckResult)


# ===========================================================================
#  REAL NLP — SECTION 1: LANGUAGE DETECTION
#  Approach: Unicode block counting + script-ratio analysis + Hinglish markers
#  This correctly identifies 8 Indian scripts without any external library.
# ===========================================================================

SCRIPT_TO_LANG: Dict[str, str] = {
    "Devanagari": "Hindi/Marathi",
    "Bengali": "Bengali",
    "Gurmukhi": "Punjabi",
    "Gujarati": "Gujarati",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
}

# Romanized Hindi/Hinglish morphological markers (common word stems)
HINGLISH_MARKERS = [
    "mera", "meri", "mujhe", "mere", "kya", "kaise", "paise", "paisa",
    "police", "dhoka", "shaadi", "naukri", "salary", "company", "account",
    "boss", "ghar", "rupee", "wala", "diya", "nahi", "nahi hai", "nahin",
    "bhi", "aur", "lekin", "isliye", "kyunki", "bahut", "zyada", "abhi",
    "phir", "sirf", "jab", "tab", "kab", "yahan", "wahan", "hoga", "gaya",
]

# Exact-form negation words (English + Hindi romanized + Devanagari)
NEGATION_TOKENS = {
    "not", "no", "never", "without", "nor", "neither",
    "nahi", "nahin", "nahi hai", "mat", "mत", "bina",
    "نہیں",  # Urdu
}
NEGATION_UNICODE = {"नहीं", "न", "मत", "नहीं है", "बिना"}


def _count_scripts(text: str) -> Counter:
    """Count characters per Unicode script block."""
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["Bengali"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["Gurmukhi"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["Gujarati"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["Tamil"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["Telugu"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["Kannada"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["Malayalam"] += 1
        elif ch.isalpha() and cp < 128:
            counts["Latin"] += 1
    return counts


def detect_language(text: str, hint: Optional[str] = None) -> str:
    """
    Real language detection using Unicode script analysis.
    Returns human-readable language name (not ISO code).
    """
    if hint:
        return f"user-hint:{hint}"

    script_counts = _count_scripts(text)
    latin = script_counts.pop("Latin", 0)
    indic_total = sum(script_counts.values())
    total = indic_total + latin

    if total == 0:
        return "Unknown"

    if indic_total > 0:
        dominant_script = max(script_counts, key=lambda k: script_counts[k])
        lang = SCRIPT_TO_LANG.get(dominant_script, dominant_script)
        indic_ratio = indic_total / total
        if indic_ratio > 0.55:
            return lang
        if latin > 0:
            return f"Mixed {lang}+Latin"

    # Hinglish detection: check for characteristic word stems
    lower = text.lower()
    hinglish_hits = sum(
        1 for m in HINGLISH_MARKERS
        if re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", lower)
    )
    if hinglish_hits >= 2:
        return "Hinglish"

    return "English"


DOMAIN_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "fundamental_rights",
        "label": "Fundamental Rights / Constitutional concern",
        "explanation": (
            "The issue may involve equality, liberty, discrimination, free speech, "
            "life/personal liberty, exploitation, or access to constitutional remedies."
        ),
        "doc": (
            # English — specific and dense
            "fundamental rights constitutional law article 14 15 19 21 22 23 24 25 32 "
            "equality before law equal protection non discrimination religion caste gender "
            "speech expression protest assembly life liberty privacy dignity untouchability "
            "forced labour child labour reservation writ petition habeas corpus state action "
            "arbitrary detention illegal arrest minority rights human rights violation "
            "fundamental right violated constitutional guarantee life and liberty "
            # Hindi/Hinglish
            "maulik adhikar samvidhan samanta dharma jati ling bhedbhav giraftaar hirasat "
            "jeevan swatantrata nijata garima jabran mazduri baal mazduri aarakshan anuchhed "
            # Devanagari
            "मौलिक अधिकार संविधान समानता भेदभाव धर्म जाति लिंग अभिव्यक्ति "
            "गिरफ्तार हिरासत जीवन स्वतंत्रता निजता गरिमा जबरन मजदूरी बाल मजदूरी "
            # Tamil
            "அடிப்படை உரிமைகள் அரசியலமைப்பு சட்டம் சமத்துவம் பாகுபாடு "
            "சாதி மதம் பாலினம் கைது தடுப்பு கட்டாய உழைப்பு "
            # Telugu
            "ప్రాథమిక హక్కులు రాజ్యాంగం సమానత్వం వివక్ష కులం మతం లింగం "
            # Bengali
            "মৌলিক অধিকার সংবিধান সমতা বৈষম্য জাত ধর্ম লিঙ্গ গ্রেপ্তার "
        ),
        "weight": 1.2,
    },
    {
        "id": "criminal_law",
        "label": "Criminal law / Public safety concern",
        "explanation": (
            "The issue may involve violence, threats, theft, assault, harassment, "
            "extortion, stalking, or other criminal concerns under Indian criminal law."
        ),
        "doc": (
            # English
            "assault attack threat kill murder rape sexual assault harassment stalking "
            "blackmail extortion theft robbery fraud cheating forgery kidnap abuse "
            "violence weapon physical harm beaten attacked threatening criminal case BNS BNSS "
            "FIR complaint arrest criminal charges mob attack street crime threatening calls "
            "person threatening physically hitting punching knife weapon attacked "
            # Hindi/Hinglish
            "maar peet dhamki hatya balatkar chhedchhad chori loot dhokha blackmail "
            "giraftaar darr dhamki maar diya pitai ki "
            # Devanagari
            "मार पिटाई धमकी हत्या बलात्कार छेड़छाड़ चोरी लूट धोखा "
            "ब्लैकमेल गिरफ्तारी एफआईआर शारीरिक हमला "
            # Tamil
            "தாக்குதல் மிரட்டல் திருட்டு கொலை பாலியல் துன்புறுத்தல் "
            # Telugu
            "దాడి బెదిరింపు దొంగతనం హత్య లైంగిక వేధింపు "
            # Bengali
            "আক্রমণ হুমকি চুরি হত্যা ধর্ষণ হয়রানি "
        ),
        "weight": 1.15,
    },
    {
        "id": "cyber_it",
        "label": "Cybercrime / Information Technology concern",
        "explanation": (
            "The issue may involve online fraud, account hacking, OTP/UPI misuse, "
            "cyber harassment, data privacy, impersonation, or digital evidence."
        ),
        "doc": (
            # English — OTP/UPI/cyber terms repeated for higher weight
            "cyber crime online fraud hacked hack OTP one-time-password UPI payment scam "
            "fake profile instagram facebook whatsapp telegram email password account qr code "
            "digital payment link loan app crypto sim swap data leak morphed photo "
            "nude photo revenge porn sextortion cyberbullying internet identity theft "
            "social media fraud impersonation unauthorized access data breach "
            "account hacked phone fraud otp shared upi transaction unknown "
            "internet money lost cyber complaint 1930 cybercrime.gov.in "
            # Hindi/Hinglish
            "online thagi OTP fraud UPI hack fake profile cyber crime internet "
            "whatsapp fraud account hack paisa gaya bina OTP ke UPI se "
            "otp share kar liya online payment gaya phone se paisa gaya "
            # Devanagari
            "ऑनलाइन धोखाधड़ी ओटीपी यूपीआई हैक साइबर अपराध फर्जी प्रोफाइल "
            "इंटरनेट खाता हैक पासवर्ड ऑनलाइन ठगी "
            # Tamil
            "ஆன்லைன் மோசடி ஒடிபி யூபிஐ ஹேக் சைபர் கிரைம் போலி "
            # Telugu
            "ఆన్‌లైన్ మోసం ఓటీపీ యూపీఐ హ్యాక్ సైబర్ "
            # Bengali
            "অনলাইন প্রতারণা ওটিপি ইউপিআই হ্যাক সাইবার জালিয়াতি "
        ),
        "weight": 1.2,
    },
    {
        "id": "consumer",
        "label": "Consumer protection / Service or product dispute",
        "explanation": (
            "The issue may involve defective goods, poor service, refund denial, "
            "misleading advertisements, e-commerce disputes, or unfair trade practices."
        ),
        "doc": (
            # English
            "consumer complaint refund replacement warranty guarantee defective damaged product "
            "poor service bad service service center ecommerce delivery amazon flipkart "
            "bill invoice seller shop overcharged misleading advertisement subscription "
            "product return defective goods quality dispute consumer forum consumer court "
            "National Consumer Helpline edaakhil product not delivered "
            # Hindi/Hinglish
            "consumer complaint refund wapas warranty kharab saman service center "
            "delivery nahi aayi bill invoice dukaan zyada charge kiya "
            # Devanagari
            "उपभोक्ता शिकायत रिफंड वारंटी खराब सामान सेवा केंद्र डिलीवरी "
            "बिल दुकान अधिक शुल्क भ्रामक विज्ञापन "
            # Tamil
            "பண்ட குறைபாடு சேவை குறைபாடு பணம் திரும்ப வாரண்டி "
            # Telugu
            "వినియోగదారు ఫిర్యాదు రీఫండ్ వారంటీ లోపభూయిష్ట "
        ),
        "weight": 1.0,
    },
    {
        "id": "labour_employment",
        "label": "Labour / Employment concern",
        "explanation": (
            "The issue may involve unpaid salary, workplace harassment, unsafe work, "
            "illegal termination, benefits, discrimination at work, or working conditions."
        ),
        "doc": (
            # English
            "salary wages unpaid termination fired layoff workplace employer employee "
            "boss hr human resources PF ESI provident fund gratuity overtime notice period "
            "resignation employment illegal dismissal wrongful termination salary dues "
            "maternity leave POSH sexual harassment at work salary not paid salary not given "
            "company not paying salary 3 months salary withheld increment due "
            # Hindi/Hinglish
            "salary nahi mili naukri se nikaala boss HR company mazdoor "
            "vetan tankhwah job termination 3 mahine salary nahi di "
            "salary nahi de raha HR jawab nahi de raha "
            # Devanagari
            "वेतन तनख्वाह नौकरी निकाल दिया कंपनी मालिक एचआर ओवरटाइम "
            "श्रम न्यायालय कार्यस्थल उत्पीड़न प्रोविडेंट फंड "
            # Tamil
            "சம்பளம் வேலை நீக்கம் நிறுவனம் பணியாளர் "
            # Telugu
            "జీతం ఉద్యోగం కంపెనీ తొలగింపు "
            # Bengali
            "বেতন চাকরি বরখাস্ত কোম্পানি "
        ),
        "weight": 1.0,
    },
    {
        "id": "civil_property",
        "label": "Civil / Property / Contract concern",
        "explanation": (
            "The issue may involve rent, property ownership, possession, contracts, "
            "loans between individuals, recovery of money, or civil disputes."
        ),
        "doc": (
            # English
            "property land house flat rent tenant landlord lease agreement contract "
            "possession ownership partition inheritance will sale deed registry "
            "boundary encroachment construction civil dispute eviction notice "
            "security deposit rental agreement property dispute transfer deed "
            "landlord not returning deposit flat vacated "
            # Hindi/Hinglish
            "zameen makaan kiraya kirayedar makan malik samjhauta kabja vasiyat "
            "deposit wapas nahi kiya flat khali kiya "
            # Devanagari
            "जमीन मकान किराया किरायेदार मकान मालिक समझौता कब्जा वसीयत "
            "बिक्री पत्र रजिस्ट्री जमानत "
            # Tamil
            "நிலம் வீடு வாடகை குத்தகை ஒப்பந்தம் "
            # Telugu
            "భూమి ఇల్లు అద్దె ఒప్పందం ఆస్తి వివాదం "
            # Bengali
            "জমি বাড়ি ভাড়া চুক্তি "
        ),
        "weight": 1.0,
    },
    {
        "id": "banking_finance",
        "label": "Banking / Financial concern",
        "explanation": (
            "The issue may involve bank transactions, loans, unauthorized debits, "
            "credit cards, recovery agents, insurance, or financial fraud."
        ),
        "doc": (
            # English
            "bank account unauthorized debit transaction credit card debit card loan "
            "EMI recovery agent insurance ATM NEFT RTGS IMPS banking "
            "chargeback RBI NBFC wallet KYC aadhaar banking fraud "
            "bank not refunding bank statement passbook account blocked cheque bounce "
            "bank ombudsman loan recovery agent bank complaint "
            # Hindi/Hinglish
            "bank account se paisa kata loan EMI recovery agent bima ATM "
            "bank mein paisa nahi aaya bank se refund nahi mila "
            # Devanagari
            "बैंक खाता लेनदेन लोन ईएमआई रिकवरी एजेंट बीमा एटीएम "
            "अनाधिकृत लेनदेन आरबीआई बैंक में पैसे "
            # Tamil
            "வங்கி கணக்கு கடன் பணம் "
            # Telugu
            "బ్యాంకు లోన్ డబ్బు ఖాతా "
            # Bengali
            "ব্যাংক ঋণ টাকা অ্যাকাউন্ট "
        ),
        "weight": 1.05,
    },
    {
        "id": "education",
        "label": "Education / Institution concern",
        "explanation": (
            "The issue may involve school/college admissions, fees, exams, certificates, "
            "discrimination, ragging, or institutional action."
        ),
        "doc": (
            # English
            "school college university student admission fees exam marksheet degree "
            "certificate scholarship teacher principal ragging hostel attendance "
            "education institution result academic discrimination TC transfer "
            "certificate withheld fee refund UGC AICTE RTE "
            # Hindi/Hinglish
            "school college student fee exam certificate scholarship ragging "
            "vidyalaya vishwavidyalay "
            # Devanagari
            "विद्यालय स्कूल कॉलेज छात्र शुल्क परीक्षा प्रमाणपत्र छात्रवृत्ति रैगिंग "
        ),
        "weight": 1.0,
    },
    {
        "id": "family_personal",
        "label": "Family / Personal law or relationship concern",
        "explanation": (
            "The issue may involve marriage, divorce, maintenance, custody, domestic "
            "violence, dowry, in-laws, elder abuse, or relationship safety."
        ),
        "doc": (
            # English
            "marriage husband wife divorce maintenance alimony custody child custody "
            "domestic violence dowry in-laws family inheritance elder abuse "
            "relationship abuse spousal abuse separation matrimonial marital "
            # Hindi/Hinglish
            "shaadi pati patni talaq dahej ghar mein hinsa sasural custody "
            "bachche ki custody maintenance bharan poshan "
            # Devanagari
            "शादी पति पत्नी तलाक दहेज घरेलू हिंसा ससुराल बच्चे की कस्टडी भरण पोषण "
            # Tamil
            "திருமணம் கணவர் மனைவி விவாகரத்து வரதட்சணை குடும்ப வன்முறை "
            # Telugu
            "పెళ్లి భర్త భార్య విడాకులు కట్నం గృహ హింస "
            # Bengali
            "বিয়ে স্বামী স্ত্রী তালাক পণ গার্হস্থ্য হিংসা "
            # Marathi
            "लग्न पती पत्नी घटस्फोट हुंडा घरगुती हिंसा "
        ),
        "weight": 1.05,
    },
    {
        "id": "governance_admin",
        "label": "Public administration / Governance concern",
        "explanation": (
            "The issue may involve police response, public services, corruption, "
            "government offices, welfare benefits, documents, or official inaction."
        ),
        "doc": (
            # English
            "government municipal panchayat police refused police not registering FIR "
            "police inaction police not helping corruption bribe demanded ration aadhaar "
            "PAN card passport voter public service government office official delay "
            "certificate license electricity water board RTI grievance official misconduct "
            "government inaction FIR not registered police refusing "
            # Hindi/Hinglish
            "sarkar police rishwat bhrashtachar RTI sarkari daftar certificate nahi mila "
            "FIR lene se mana police ne FIR nahi li rishwat maang rahi "
            # Devanagari
            "सरकार नगर निगम पंचायत पुलिस भ्रष्टाचार रिश्वत आधार राशन "
            "सरकारी दफ्तर प्रमाणपत्र आरटीआई एफआईआर दर्ज नहीं "
            # Tamil
            "அரசு காவல்துறை லஞ்சம் சான்றிதழ் "
            # Telugu
            "ప్రభుత్వం పోలీసు లంచం సర్టిఫికేట్ "
            # Bengali
            "সরকার পুলিশ ঘুষ সার্টিফিকেট "
        ),
        "weight": 1.05,
    },
    {
        "id": "non_legal_social",
        "label": "Non-legal / Social / Emotional / Informational issue",
        "explanation": (
            "The input may describe stress, interpersonal conflict, general advice needs, "
            "or incomplete facts without a clear legal/civic issue."
        ),
        "doc": (
            "sad stress depressed anxiety confused relationship problem career advice "
            "what should i do help me decide mental health lonely friendship motivation "
            "feeling upset emotional problem personal advice general question informational "
        ),
        "weight": 0.75,
    },
]


_domain_docs = [d["doc"] for d in DOMAIN_CORPUS]
_domain_ids = [d["id"] for d in DOMAIN_CORPUS]
_domain_labels = {d["id"]: d["label"] for d in DOMAIN_CORPUS}
_domain_explanations = {d["id"]: d["explanation"] for d in DOMAIN_CORPUS}
_domain_weights = {d["id"]: d["weight"] for d in DOMAIN_CORPUS}

# Char n-gram vectorizer: script-agnostic, works on any Unicode text
_vec_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    sublinear_tf=True,
    max_features=80_000,
    strip_accents=None,  # Preserve Devanagari/Tamil diacritics
)

# Word n-gram vectorizer: captures English and Romanized Hindi phrases
_vec_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=30_000,
)

_X_char = _vec_char.fit_transform(_domain_docs)
_X_word = _vec_word.fit_transform(_domain_docs)

# Domain weight vector for boosting/suppressing domains
_domain_weight_vec = np.array([_domain_weights[d] for d in _domain_ids])


# English negation patterns
_NEG_PATTERN_EN = re.compile(
    r"\b(not|no|never|without|nor|neither|dont|don't|isn't|wasn't|hasn't"
    r"|haven't|didn't|cannot|can't|won't|wouldn't|shouldn't)\b",
    re.IGNORECASE,
)
# Hindi/Hinglish romanized negation
_NEG_PATTERN_HI_ROMAN = re.compile(
    r"\b(nahi|nahin|nahi\s+hai|mat|bina)\b", re.IGNORECASE
)
# Devanagari negation characters
_NEG_PATTERN_DEVA = re.compile(r"(नहीं|न\s|मत|बिना)")


def _contains_negation_near(text: str, start: int, window_chars: int = 60) -> bool:
    """Return True if a negation word appears within window_chars before start."""
    snippet = text[max(0, start - window_chars): start]
    return bool(
        _NEG_PATTERN_EN.search(snippet)
        or _NEG_PATTERN_HI_ROMAN.search(snippet)
        or _NEG_PATTERN_DEVA.search(snippet)
    )



# (phrase, domain_id, bonus_score)
PHRASE_SIGNALS: List[Tuple[str, str, float]] = [
    # Criminal
    ("domestic violence", "family_personal", 18.0),
    ("sexual assault", "criminal_law", 22.0),
    ("sexual harassment", "criminal_law", 18.0),
    ("threatening messages", "criminal_law", 16.0),
    ("physical assault", "criminal_law", 18.0),
    ("blackmail me", "criminal_law", 20.0),
    ("file fir", "criminal_law", 15.0),
    ("police refused", "governance_admin", 18.0),
    ("police not registering", "governance_admin", 18.0),
    # Cyber
    ("otp fraud", "cyber_it", 22.0),
    ("upi fraud", "cyber_it", 22.0),
    ("account hacked", "cyber_it", 22.0),
    ("unauthorized transaction", "cyber_it", 20.0),
    ("online scam", "cyber_it", 18.0),
    ("phishing link", "cyber_it", 18.0),
    ("fake profile", "cyber_it", 16.0),
    ("sim swap", "cyber_it", 20.0),
    ("morphed photo", "cyber_it", 20.0),
    ("revenge porn", "cyber_it", 22.0),
    # Labour
    ("salary not paid", "labour_employment", 22.0),
    ("wrongful termination", "labour_employment", 20.0),
    ("illegal termination", "labour_employment", 20.0),
    ("notice period", "labour_employment", 14.0),
    ("salary dues", "labour_employment", 18.0),
    ("pf not deposited", "labour_employment", 18.0),
    ("workplace harassment", "labour_employment", 18.0),
    # Property
    ("security deposit", "civil_property", 20.0),
    ("rent agreement", "civil_property", 18.0),
    ("illegal eviction", "civil_property", 20.0),
    ("property dispute", "civil_property", 16.0),
    ("sale deed", "civil_property", 16.0),
    # Banking
    ("money deducted", "banking_finance", 20.0),
    ("bank fraud", "banking_finance", 20.0),
    ("loan recovery", "banking_finance", 16.0),
    ("credit card fraud", "banking_finance", 20.0),
    ("upi payment failed", "banking_finance", 16.0),
    # Consumer
    ("product defective", "consumer", 18.0),
    ("refund not given", "consumer", 18.0),
    ("warranty claim", "consumer", 16.0),
    ("misleading advertisement", "consumer", 16.0),
    # Governance
    ("bribe demanded", "governance_admin", 20.0),
    ("rti application", "governance_admin", 18.0),
    ("certificate delayed", "governance_admin", 16.0),
    ("ration card", "governance_admin", 14.0),
    # Constitutional
    ("fundamental rights", "fundamental_rights", 20.0),
    ("constitutional rights", "fundamental_rights", 18.0),
    ("right to life", "fundamental_rights", 18.0),
    ("right to equality", "fundamental_rights", 16.0),
    # Hindi/Hinglish phrase signals
    ("salary nahi mili", "labour_employment", 22.0),
    ("paisa nahi wapas", "banking_finance", 18.0),
    ("otp share kar liya", "cyber_it", 22.0),
    ("upi se paisa gaya", "cyber_it", 20.0),
    ("kiraaya wapas nahi", "civil_property", 20.0),
    ("dahej ki maang", "family_personal", 22.0),
    ("ghar se nikala", "family_personal", 16.0),
    ("dhamki de raha", "criminal_law", 20.0),
]


def _compute_phrase_bonus(text_lower: str) -> Dict[str, float]:
   
    bonuses: Dict[str, float] = {}
    for phrase, domain_id, bonus in PHRASE_SIGNALS:
        m = re.search(re.escape(phrase), text_lower)
        if m and not _contains_negation_near(text_lower, m.start()):
            bonuses[domain_id] = bonuses.get(domain_id, 0.0) + bonus
    return bonuses



LEGAL_DOMAIN_IDS = {
    "fundamental_rights", "criminal_law", "cyber_it", "consumer",
    "labour_employment", "civil_property", "banking_finance",
    "education", "family_personal", "governance_admin",
}

DISPLAY_SIGNALS: Dict[str, List[str]] = {
    "fundamental_rights": ["fundamental right", "constitutional", "article 14", "article 21", "discrimination", "bhedbhav", "भेदभाव", "discrimination", "equality", "caste", "religion"],
    "criminal_law": ["assault", "threat", "murder", "rape", "harassment", "theft", "blackmail", "extortion", "FIR", "maar", "dhamki", "मार", "धमकी", "हत्या", "चोरी"],
    "cyber_it": ["OTP", "UPI", "hacked", "phishing", "fake profile", "cybercrime", "online fraud", "ऑनलाइन", "साइबर", "ओटीपी", "यूपीआई"],
    "consumer": ["refund", "warranty", "defective", "delivery", "consumer", "ecommerce", "रिफंड", "वारंटी", "खराब"],
    "labour_employment": ["salary", "wages", "fired", "termination", "HR", "PF", "तनख्वाह", "वेतन", "नौकरी", "निकाल"],
    "civil_property": ["rent", "property", "landlord", "deposit", "lease", "agreement", "किराया", "जमीन", "मकान", "कब्जा"],
    "banking_finance": ["bank", "loan", "transaction", "EMI", "credit card", "RBI", "बैंक", "लोन", "ईएमआई"],
    "education": ["school", "college", "fees", "exam", "ragging", "certificate", "स्कूल", "कॉलेज", "फीस", "परीक्षा"],
    "family_personal": ["marriage", "divorce", "domestic violence", "dowry", "custody", "शादी", "तलाक", "दहेज", "घरेलू हिंसा"],
    "governance_admin": ["government", "police", "bribe", "corruption", "RTI", "certificate", "सरकार", "पुलिस", "रिश्वत", "भ्रष्टाचार"],
    "non_legal_social": ["stress", "sad", "depressed", "career", "advice", "परेशान", "तनाव"],
}


def _find_display_signals(text_lower: str, domain_id: str) -> List[str]:
    """Find which display-signal words are present (for UI signal chips)."""
    signals = DISPLAY_SIGNALS.get(domain_id, [])
    found = []
    for s in signals:
        if s.lower() in text_lower:
            found.append(s)
    return found[:10]


def classify_domains(
    text: str,
) -> Tuple[List[MatchedDomain], int, bool]:
   
    text_lower = text.lower()

    # --- Step 1: TF-IDF semantic scores ---
    q_char = _vec_char.transform([text])
    q_word = _vec_word.transform([text])

    sim_char = cosine_similarity(q_char, _X_char)[0]  # shape: (n_domains,)
    sim_word = cosine_similarity(q_word, _X_word)[0]

    # Weighted combination: char n-grams get more weight for Indic scripts
    sim_combined = sim_char * 0.6 + sim_word * 0.4

   
    base_scores = np.minimum(sim_combined / 0.55, 1.0) * 100.0

    # --- Step 3: Phrase bonus ---
    phrase_bonuses = _compute_phrase_bonus(text_lower)
    for i, domain_id in enumerate(_domain_ids):
        base_scores[i] += phrase_bonuses.get(domain_id, 0.0)

    # --- Step 4: Domain weight multiplier ---
    base_scores = base_scores * _domain_weight_vec

    # --- Step 5: Build result list (threshold: score > 25) ---
    threshold = 25.0
    scored = sorted(
        [(score, i) for i, score in enumerate(base_scores) if score >= threshold],
        reverse=True,
    )

    if not scored:
        return [], 22, True

    domains: List[MatchedDomain] = []
    for score, idx in scored[:4]:
        domain_id = _domain_ids[idx]
        display_sigs = _find_display_signals(text_lower, domain_id)
        # Also add any phrase signals that matched
        for phrase, pid, _ in PHRASE_SIGNALS:
            if pid == domain_id and phrase in text_lower:
                if phrase not in display_sigs:
                    display_sigs.append(phrase)
        domains.append(
            MatchedDomain(
                id=domain_id,
                label=_domain_labels[domain_id],
                score=round(min(score, 99.9), 1),
                explanation=_domain_explanations[domain_id],
                matched_signals=unique_keep_order(display_sigs)[:12],
            )
        )

    # --- Step 6: Calibrated confidence score ---
    top_score = scored[0][0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    separation = max(0.0, top_score - second_score)

    # Word count: longer text gives more signal
    word_count = len(text.split())
    length_bonus = min(15.0, word_count * 1.1)

    # Evidence indicators boost confidence
    evidence_terms = [
        "screenshot", "receipt", "invoice", "message", "recording",
        "bank statement", "agreement", "proof", "witness",
        "गवाह", "रसीद", "स्क्रीनशॉट",
    ]
    evidence_bonus = 6.0 if any(t in text_lower for t in evidence_terms) else 0.0

    # Separation bonus: clear winner vs ambiguous
    separation_bonus = min(10.0, separation / 5.0)

    raw_conf = (
        top_score * 0.55
        + length_bonus
        + separation_bonus
        + evidence_bonus
    )
    confidence = int(np.clip(raw_conf, 20, 95))
    uncertainty = confidence < LOW_CONFIDENCE_THRESHOLD

    return domains, confidence, uncertainty


_MONEY_PATTERNS = [
    re.compile(r"₹\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|lakhs|crore|crores|thousand))?", re.IGNORECASE),
    re.compile(r"\brs\.?\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|lakhs|crore|crores|thousand))?", re.IGNORECASE),
    re.compile(r"\binr\s?[\d,]+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\b[\d,]+(?:\.\d+)?\s?(?:rupees|rupee|lakh|lakhs|crore|crores)\b", re.IGNORECASE),
]

_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"),
    re.compile(r"\b(?:today|yesterday|tomorrow|last\s+(?:night|week|month|year)|this\s+(?:week|month|year))\b", re.IGNORECASE),
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s?(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"\b(?:आज|कल|परसों|पिछले\s+(?:सप्ताह|महीने|साल)|इस\s+(?:सप्ताह|महीने))\b"),
    re.compile(r"\b\d+\s+(?:days?|weeks?|months?|years?|din|mahine|saal|hafte)\s+(?:ago|back|se|pehle)\b", re.IGNORECASE),
]

_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
_AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

_EVIDENCE_TERMS = [
    "screenshot", "recording", "video", "audio", "photo", "receipt",
    "invoice", "bill", "chat", "message", "email", "document", "agreement",
    "witness", "transaction id", "utr", "bank statement", "call log",
    "स्क्रीनशॉट", "रिकॉर्डिंग", "वीडियो", "फोटो", "रसीद", "गवाह",
]

_SENSITIVE_TERMS = [
    "rape", "sexual assault", "minor", "child abuse", "suicide",
    "self harm", "self-harm", "kill myself", "domestic violence",
    "बलात्कार", "आत्महत्या", "बच्चा", "नाबालिग", "घरेलू हिंसा",
]

_ORG_KEYWORDS = [
    "bank", "school", "college", "company", "police", "hospital",
    "municipal", "university", "court", "insurance", "ngo",
]


def extract_entities(text: str) -> EntityExtraction:

    money: List[str] = []
    for pat in _MONEY_PATTERNS:
        money.extend(pat.findall(text))

    dates: List[str] = []
    for pat in _DATE_PATTERNS:
        dates.extend(pat.findall(text))

    # Location candidates: capitalized sequences not in stopwords
    _stop = {
        "I", "My", "The", "A", "An", "He", "She", "We", "They",
        "In", "At", "On", "By", "For", "From", "With", "About",
        "Nyaya", "AI", "Hindi", "English",
    }
    cap_candidates = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b", text)
    locations = [c for c in cap_candidates if c not in _stop][:8]

    text_lower = text.lower()
    organizations = [kw for kw in _ORG_KEYWORDS if kw in text_lower]

    contact_present = bool(_phone_or_email(text))

    evidence = [t for t in _EVIDENCE_TERMS if t.lower() in text_lower]
    sensitive = [t for t in _SENSITIVE_TERMS if t.lower() in text_lower]

    return EntityExtraction(
        money_amounts=unique_keep_order(money)[:10],
        dates_or_time_refs=unique_keep_order(dates)[:10],
        locations=unique_keep_order(locations)[:8],
        organizations=unique_keep_order(organizations)[:8],
        contact_indicators_present=contact_present,
        evidence_indicators=evidence[:10],
        sensitive_terms=sensitive[:10],
    )


def _phone_or_email(text: str) -> Optional[re.Match]:
    return _EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text)



HIGH_RISK_SIGNALS = {
    "violence/threat": [
        "kill", "murder", "attack", "weapon", "assault", "beaten", "rape",
        "sexual assault", "kidnap", "maar dalo", "jaan se maaro",
        "धमकी", "हत्या", "बलात्कार", "मार डाल",
    ],
    "self-harm": ["suicide", "self harm", "self-harm", "kill myself", "आत्महत्या"],
    "minor/child safety": ["minor", "child abuse", "under 18", "नाबालिग", "बच्चे के साथ"],
    "active extortion/blackmail": [
        "blackmail", "extortion", "sextortion", "pay or", "leak my photo", "ब्लैकमेल",
    ],
    "domestic violence": [
        "domestic violence", "ghar mein maar", "ghar mein hinsa",
        "घरेलू हिंसा", "dowry violence",
    ],
}

MEDIUM_RISK_SIGNALS = {
    "financial loss": [
        "lost money", "money deducted", "fraud", "scam", "loan", "unauthorized transaction",
        "₹", "upi fraud", "otp fraud", "पैसे गए", "रुपये",
    ],
    "repeated harassment": [
        "harassment", "stalking", "repeated", "again and again", "daily",
        "pareshan", "baar baar", "परेशान", "बार बार",
    ],
    "job/income impact": [
        "salary unpaid", "unpaid salary", "fired", "termination",
        "वेतन नहीं", "नौकरी गई",
    ],
    "official inaction": [
        "police refused", "not taking complaint", "government delay",
        "bribe demanded", "रिश्वत", "पुलिस ने मना",
    ],
}


def _contains_any(text_lower: str, terms: List[str]) -> bool:
    return any(t.lower() in text_lower for t in terms)


def analyze_risk(
    text: str, entities: EntityExtraction
) -> Tuple[RiskLevel, List[str]]:
    lower = text.lower()
    reasons: List[str] = []

    for label, terms in HIGH_RISK_SIGNALS.items():
        if _contains_any(lower, terms):
            reasons.append(f"High-risk indicator detected: {label}.")

    max_money = _estimate_max_money(entities.money_amounts)
    if max_money >= 100_000:
        reasons.append("Large financial amount mentioned; potential impact may be high.")
    elif max_money >= 10_000:
        reasons.append("Meaningful financial amount mentioned; potential impact may be medium.")

    for label, terms in MEDIUM_RISK_SIGNALS.items():
        if _contains_any(lower, terms):
            reasons.append(f"Medium-risk indicator detected: {label}.")

    if any(r.startswith("High-risk") for r in reasons) or max_money >= 100_000:
        return RiskLevel.HIGH, unique_keep_order(reasons)[:8]
    if reasons:
        return RiskLevel.MEDIUM, unique_keep_order(reasons)[:8]
    if len(text.split()) < 5:
        return RiskLevel.UNKNOWN, ["Input is too brief to estimate impact reliably."]
    return RiskLevel.LOW, ["No immediate high-risk indicators were detected."]


def _estimate_max_money(amounts: List[str]) -> float:
    max_val = 0.0
    for amount in amounts:
        a = amount.lower().replace(",", "")
        nums = re.findall(r"\d+(?:\.\d+)?", a)
        if not nums:
            continue
        val = float(nums[0])
        if "crore" in a:
            val *= 10_000_000
        elif "lakh" in a:
            val *= 100_000
        max_val = max(max_val, val)
    return max_val


RIGHTS_MAP: Dict[str, List[RightsMapping]] = {
    "fundamental_rights": [
        RightsMapping(title="Equality before law and equal protection", reference="Article 14, Constitution of India", relevance="May be relevant where state action or public authority conduct appears unequal, arbitrary, or discriminatory.", confidence="context-dependent"),
        RightsMapping(title="Non-discrimination by the State", reference="Article 15, Constitution of India", relevance="May be relevant if the facts indicate discrimination on protected grounds such as religion, race, caste, sex, or place of birth.", confidence="context-dependent"),
        RightsMapping(title="Protection of life and personal liberty", reference="Article 21, Constitution of India", relevance="May be relevant for serious threats to dignity, privacy, safety, liberty, or humane treatment, especially involving state action.", confidence="context-dependent"),
        RightsMapping(title="Constitutional remedies", reference="Article 32, Constitution of India", relevance="Educationally relevant where a genuine Fundamental Rights violation by State action is alleged; professional legal advice is needed.", confidence="context-dependent"),
    ],
    "criminal_law": [
        RightsMapping(title="Criminal law domain", reference="Bharatiya Nyaya Sanhita, 2023; Bharatiya Nagarik Suraksha Sanhita, 2023; Bharatiya Sakshya Adhiniyam, 2023", relevance="May be relevant where alleged facts include threats, assault, theft, cheating, harassment, extortion, or violence.", confidence="domain-level only")
    ],
    "cyber_it": [
        RightsMapping(title="Cybercrime and digital safety domain", reference="Information Technology Act, 2000 and related cybercrime processes", relevance="May be relevant for hacking, OTP/UPI fraud, online impersonation, cyber harassment, privacy/data misuse, or digital extortion.", confidence="domain-level only")
    ],
    "consumer": [
        RightsMapping(title="Consumer protection domain", reference="Consumer Protection Act, 2019", relevance="May be relevant for defective goods, deficient services, refund/warranty disputes, unfair trade practices, or misleading advertisements.", confidence="domain-level only")
    ],
    "labour_employment": [
        RightsMapping(title="Labour and employment domain", reference="Indian labour/employment laws and applicable service rules/contracts", relevance="May be relevant for unpaid wages, workplace harassment, termination, benefits, unsafe working conditions, or employment discrimination.", confidence="domain-level only")
    ],
    "civil_property": [
        RightsMapping(title="Civil/property/contract domain", reference="Civil law, property law, contract law, tenancy and succession principles as applicable", relevance="May be relevant for ownership, rent, agreements, possession, inheritance, deposits, and recovery disputes.", confidence="domain-level only")
    ],
    "banking_finance": [
        RightsMapping(title="Banking and financial services domain", reference="RBI-regulated banking/payment framework and financial consumer grievance mechanisms", relevance="May be relevant for unauthorized transactions, loan/recovery issues, account problems, cards, insurance, or financial fraud.", confidence="domain-level only")
    ],
    "education": [
        RightsMapping(title="Education and institutional governance domain", reference="Education regulations, institutional rules, anti-ragging norms, and applicable statutory frameworks", relevance="May be relevant for admissions, fees, exams, certificates, ragging, discrimination, or institutional grievances.", confidence="domain-level only")
    ],
    "family_personal": [
        RightsMapping(title="Family and personal law domain", reference="Applicable personal laws, family law protections, and domestic violence protections", relevance="May be relevant for domestic violence, marriage, divorce, maintenance, custody, dowry-related concerns, or elder abuse.", confidence="domain-level only")
    ],
    "governance_admin": [
        RightsMapping(title="Public administration and governance domain", reference="Administrative grievance systems, anti-corruption frameworks, RTI/public service mechanisms as applicable", relevance="May be relevant for official delay, refusal, corruption demands, police inaction, public service denial, or document/service grievances.", confidence="domain-level only")
    ],
}

DPSP_AND_DUTY_NOTES = [
    RightsMapping(title="Directive Principles of State Policy - welfare orientation", reference="Part IV, Constitution of India", relevance="DPSPs guide the State in areas like social justice, livelihood, education, public health, and welfare. They are generally not directly enforceable like Fundamental Rights.", confidence="educational context"),
    RightsMapping(title="Fundamental Duties - civic responsibility", reference="Article 51A, Constitution of India", relevance="Citizens are encouraged to uphold constitutional values, harmony, public property, scientific temper, and other civic duties.", confidence="educational context"),
]


def map_rights_and_laws(domains: List[MatchedDomain], confidence: int) -> List[RightsMapping]:
    if confidence < LOW_CONFIDENCE_THRESHOLD or not domains:
        return []
    mappings: List[RightsMapping] = []
    for domain in domains[:3]:
        if domain.id in RIGHTS_MAP and domain.score >= 30:
            if domain.id == "fundamental_rights":
                mappings.extend(RIGHTS_MAP[domain.id][:4])
            else:
                mappings.extend(RIGHTS_MAP[domain.id])
    civic_ids = {"fundamental_rights", "governance_admin", "education", "labour_employment"}
    if any(d.id in civic_ids for d in domains):
        mappings.extend(DPSP_AND_DUTY_NOTES)
    seen: set = set()
    out: List[RightsMapping] = []
    for item in mappings:
        key = (item.title, item.reference)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:8]


def generate_clarifying_questions(
    domains: List[MatchedDomain], confidence: int, text: str
) -> List[str]:
    questions: List[str] = []
    if confidence < LOW_CONFIDENCE_THRESHOLD or not domains:
        return [
            "What exactly happened, and who was involved?",
            "When and where did it happen?",
            "Was there money loss, threat, violence, discrimination, harassment, or official inaction?",
            "What evidence do you have — screenshots, receipts, messages, documents, or witnesses?",
        ]
    ids = {d.id for d in domains}
    if "cyber_it" in ids or "banking_finance" in ids:
        questions.append("Was any OTP, UPI PIN, password, card detail, or remote-access app involved?")
        questions.append("Do you have transaction IDs, screenshots, bank messages, or complaint reference numbers?")
    if "labour_employment" in ids:
        questions.append("Do you have an appointment letter, salary slips, attendance proof, or written HR communication?")
    if "civil_property" in ids:
        questions.append("Is there a written agreement, rent receipt, sale deed, or ownership document?")
    if "criminal_law" in ids:
        questions.append("Is there any immediate safety risk or repeated threat/harassment?")
    if "fundamental_rights" in ids:
        questions.append("Is the alleged action by a government/public authority, or by a private person/entity?")
    return questions[:4]


def primary_category(domains: List[MatchedDomain], confidence: int) -> str:
    if not domains or confidence < LOW_CONFIDENCE_THRESHOLD:
        return "uncertain / insufficient information"
    return domains[0].label


def build_summary(text: str, domains: List[MatchedDomain], risk_level: RiskLevel) -> str:
    snippet = _safe_snippet(text)
    if not domains:
        return f'The user described: "{snippet}". The input is too limited for reliable legal/civic classification.'
    return (
        f'The user appears to describe: "{snippet}". '
        f"Strongest detected category: {domains[0].label}. "
        f"Estimated impact level: {risk_level.value}."
    )


def _safe_snippet(text: str, max_len: int = 220) -> str:
    masked = _EMAIL_PATTERN.sub("[email masked]", text)
    masked = _PHONE_PATTERN.sub("[phone masked]", masked)
    masked = _AADHAAR_PATTERN.sub("[12-digit ID masked]", masked)
    return masked[:max_len - 3].rstrip() + "..." if len(masked) > max_len else masked


def safety_notes(confidence: int, domains: List[MatchedDomain]) -> List[str]:
    notes = [
        "No person or organization is declared guilty, liable, or dishonest based only on this input.",
        "Mappings are domain-level educational signals, not legal conclusions or filing instructions.",
        "Do not share OTPs, passwords, private IDs, or full account details in chat systems.",
    ]
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        notes.append("Uncertainty mode is active because the facts are incomplete or weakly matched.")
    if any(d.id == "fundamental_rights" for d in domains):
        notes.append("Fundamental Rights claims often depend on whether State/public authority action is involved.")
    return notes


def unique_keep_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        clean = str(item).strip()
        if not clean:
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def analyze_problem(text: str, language_hint: Optional[str] = None) -> AnalysisResponse:
    normalized = normalize_text(text)
    detected = detect_language(normalized, language_hint)
    entities = extract_entities(normalized)
    domains, confidence, uncertainty = classify_domains(normalized)
    risk_level, risk_reasons = analyze_risk(normalized, entities)

    # If only non-legal/social matched strongly, suppress legal mapping
    if domains and domains[0].id == "non_legal_social" and confidence < HIGH_CONFIDENCE_THRESHOLD:
        uncertainty = True
        confidence = min(confidence, 55)

    mappings = map_rights_and_laws(domains, confidence)

    uncertainty_note: Optional[str] = None
    if uncertainty:
        uncertainty_note = (
            "No reliable legal or constitutional mapping is possible from the current facts. "
            "The result is limited to general awareness and clarifying questions."
        )

    return AnalysisResponse(
        disclaimer=DISCLAIMER,
        normalized_summary=build_summary(normalized, domains, risk_level),
        detected_language=detected,
        primary_category=primary_category(domains, confidence),
        domains=domains,
        possible_rights_or_law_domains=mappings,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        confidence_score=confidence,
        uncertainty_mode=uncertainty,
        uncertainty_note=uncertainty_note,
        extracted_context=entities,
        safety_notes=safety_notes(confidence, domains),
        clarifying_questions=generate_clarifying_questions(domains, confidence, normalized),
    )


app = FastAPI(
    title="Nyaya AI Civic Awareness API",
    version="2.0.0",
    description=(
        "Multilingual civic/legal awareness analyzer for India with real NLP. "
        "Educational only; not legal advice."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def serve_frontend():
    index = Path(__file__).with_name("index.html")
    if not index.exists():
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "message": "Nyaya AI API is running. No frontend index.html found."},
        )
    return FileResponse(index)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "nyaya-ai",
        "version": "3.0.0",
        "nlp": "tfidf-semantic+phrase-context+negation-aware",
        "mode": "educational-awareness",
        "rag": {
            "knowledge_base_entries": len(rag_engine.KNOWLEDGE_BASE),
            "gemini_model": rag_engine.GEMINI_MODEL,
            "gemini_api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        },
        "research_features": {
            "self_check_on_rag_explain": True,
            "self_check_on_agentic_chat": True,
            "baseline_endpoint": "/api/baseline-explain",
            "single_eval_endpoint": "/api/evaluate",
            "batch_eval_endpoint": "/api/evaluate-suite",
            "gold_dataset_endpoint": "/api/gold-dataset",
            "gold_dataset_size": len(rag_engine.GOLD_DATASET),
            "eval_dimensions": ["grounding", "actionability", "hallucination", "relevance"],
            "eval_judges": 2,
            "inter_rater_reliability": "cohens_kappa_per_dimension",
            "hallucination_detection": "regex_extraction_vs_gold_helplines_and_sections",
            "multilingual_retrieval": True,
        },
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_api(req: AnalysisRequest) -> AnalysisResponse:
    return analyze_problem(req.text, req.language_hint)


@app.post("/api/rag-explain", response_model=RagAnalysisResponse)
def rag_explain_api(req: AnalysisRequest) -> RagAnalysisResponse:
    
    analysis = analyze_problem(req.text, req.language_hint)
    domain_ids = [d.id for d in analysis.domains]
    rag_result = rag_engine.get_rag_explanation(
        user_text=req.text,
        primary_category=analysis.primary_category,
        risk_level=analysis.risk_level.value,
        domain_ids=domain_ids,
        language_hint=req.language_hint,
    )
    sc = rag_result.pop("self_check", {})
    return RagAnalysisResponse(
        **analysis.dict(),
        **rag_result,
        self_check=SelfCheckResult(
            ok=sc.get("ok"),
            issues=sc.get("issues", []),
            revised=sc.get("revised", False),
        ),
    )


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    conversation: List[ChatMessage] = Field(default_factory=list)
    domain_hint: Optional[str] = None
    language_hint: Optional[str] = None


class AgentPlanQuery(BaseModel):
    query: str
    domain: str


class AgentTraceStep(BaseModel):
    model_config = ConfigDict(extra="ignore")
    step: str
    label: str
    reasoning: Optional[str] = None
    queries: Optional[List[AgentPlanQuery]] = None
    domain: Optional[str] = None
    found: Optional[Any] = None
    ok: Optional[bool] = None
    issues: Optional[List[str]] = None
    skipped: Optional[bool] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[RagSource]
    ai_available: bool
    disclaimer: str
    agent_trace: List[AgentTraceStep] = Field(default_factory=list)


@app.post("/api/chat", response_model=ChatResponse)
def chat_api(req: ChatRequest) -> ChatResponse:
    conversation = [{"role": m.role, "content": m.content} for m in req.conversation]
    result = rag_engine.agentic_chat_response(
        user_message=req.message,
        conversation=conversation,
        domain_hint=req.domain_hint,
    )
    return ChatResponse(
        response=result["response"],
        sources=[RagSource(**s) for s in result["sources"]],
        ai_available=result["ai_available"],
        disclaimer=DISCLAIMER,
        agent_trace=result.get("agent_trace", []),
    )


class BaselineAnalysisResponse(BaseModel):
    
    user_text: str
    primary_category: str
    risk_level: str
    ai_available: bool
    ai_explanation: str
    disclaimer: str


@app.post("/api/baseline-explain", response_model=BaselineAnalysisResponse)
def baseline_explain_api(req: AnalysisRequest) -> BaselineAnalysisResponse:
   
    analysis = analyze_problem(req.text, req.language_hint)
    baseline = rag_engine.get_baseline_explanation(
        user_text=req.text,
        primary_category=analysis.primary_category,
        risk_level=analysis.risk_level.value,
    )
    return BaselineAnalysisResponse(
        user_text=req.text,
        primary_category=analysis.primary_category,
        risk_level=analysis.risk_level.value,
        ai_available=baseline["ai_available"],
        ai_explanation=baseline["ai_explanation"],
        disclaimer=DISCLAIMER,
    )


class EvalRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    language_hint: Optional[str] = None


class EvalDimScores(BaseModel):
    grounding: Optional[float] = None
    actionability: Optional[float] = None
    hallucination: Optional[float] = None
    relevance: Optional[float] = None


class HallucinationMetric(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    extra: List[str] = []
    predicted: List[str] = []
    gold: List[str] = []


class HallucinationScores(BaseModel):
    helplines: HallucinationMetric = Field(default_factory=HallucinationMetric)
    sections: HallucinationMetric = Field(default_factory=HallucinationMetric)


class EvalResponse(BaseModel):
    
    user_text: str
    primary_category: str
    risk_level: str
    rag_explanation: str
    baseline_explanation: str
    rag_self_check: SelfCheckResult
    # Averaged scores across both judges
    rag: EvalDimScores
    baseline: EvalDimScores
    rag_total: Optional[float] = None
    baseline_total: Optional[float] = None
    winner: Optional[str] = None
    key_difference: Optional[str] = None
    # Per-judge raw scores
    judge1: Optional[dict] = None
    judge2: Optional[dict] = None
    # Inter-rater reliability for this case
    kappa: Optional[dict] = None
    # Hallucination F1 (populated when query matches a gold case)
    hallucination: Optional[dict] = None
    gold_matched: bool = False
    eval_available: bool
    disclaimer: str


@app.post("/api/evaluate", response_model=EvalResponse)
def evaluate_api(req: EvalRequest) -> EvalResponse:
    
    analysis = analyze_problem(req.text, req.language_hint)
    domain_ids = [d.id for d in analysis.domains]

    # RAG response (with self-check)
    rag_result = rag_engine.get_rag_explanation(
        user_text=req.text,
        primary_category=analysis.primary_category,
        risk_level=analysis.risk_level.value,
        domain_ids=domain_ids,
        language_hint=req.language_hint,
    )

    # Baseline response (no retrieval)
    baseline_result = rag_engine.get_baseline_explanation(
        user_text=req.text,
        primary_category=analysis.primary_category,
        risk_level=analysis.risk_level.value,
    )

    # Retrieve passages again for the evaluator's reference context
    passages = rag_engine.KnowledgeBaseRetriever(domain_ids=domain_ids, top_k=5).invoke(req.text)

    # Look up gold case (exact text match)
    gold = next((g for g in rag_engine.GOLD_DATASET if g.text == req.text), None)

    eval_result: dict = {"available": False, "rag": {}, "baseline": {}}
    if rag_result["ai_available"] and baseline_result["ai_available"]:
        eval_result = rag_engine.evaluate_rag_vs_baseline(
            user_text=req.text,
            rag_response=rag_result["ai_explanation"],
            baseline_response=baseline_result["ai_explanation"],
            passages=passages,
            gold=gold,
        )

    sc = rag_result.get("self_check", {})
    return EvalResponse(
        user_text=req.text,
        primary_category=analysis.primary_category,
        risk_level=analysis.risk_level.value,
        rag_explanation=rag_result["ai_explanation"],
        baseline_explanation=baseline_result["ai_explanation"],
        rag_self_check=SelfCheckResult(
            ok=sc.get("ok"),
            issues=sc.get("issues", []),
            revised=sc.get("revised", False),
        ),
        rag=EvalDimScores(**eval_result.get("rag", {})) if eval_result.get("available") else EvalDimScores(),
        baseline=EvalDimScores(**eval_result.get("baseline", {})) if eval_result.get("available") else EvalDimScores(),
        rag_total=eval_result.get("rag_total"),
        baseline_total=eval_result.get("baseline_total"),
        winner=eval_result.get("winner"),
        key_difference=eval_result.get("key_difference"),
        judge1=eval_result.get("judge1"),
        judge2=eval_result.get("judge2"),
        kappa=eval_result.get("kappa"),
        hallucination=eval_result.get("hallucination"),
        gold_matched=gold is not None,
        eval_available=eval_result.get("available", False),
        disclaimer=DISCLAIMER,
    )


class EvalSuiteTestCase(BaseModel):
    user_text: str
    primary_category: str = ""
    risk_level: str = "unknown"
    domain_ids: List[str] = Field(default_factory=list)


class EvalSuiteRequest(BaseModel):
    test_cases: List[EvalSuiteTestCase] = Field(..., min_length=1, max_length=50)
    auto_classify: bool = Field(
        True,
        description=(
            "If true, run TF-IDF analysis on each case to fill in "
            "primary_category, risk_level, and domain_ids automatically. "
            "Set false to use the values you provide (faster, but you must "
            "supply all three fields)."
        ),
    )


class EvalSuiteResponse(BaseModel):
    n_cases: int
    n_evaluated: int
    n_gold_matched: int = 0
    rag_means: dict
    baseline_means: dict
    kappa_means: dict = Field(default_factory=dict)
    hallucination_f1: dict = Field(default_factory=dict)
    wins: dict
    results: List[dict]
    disclaimer: str


@app.post("/api/evaluate-suite", response_model=EvalSuiteResponse)
def evaluate_suite_api(req: EvalSuiteRequest) -> EvalSuiteResponse:
    
    prepared = []
    for tc in req.test_cases:
        case = tc.dict()
        if req.auto_classify:
            analysis = analyze_problem(tc.user_text)
            case["primary_category"] = analysis.primary_category
            case["risk_level"] = analysis.risk_level.value
            case["domain_ids"] = [d.id for d in analysis.domains]
        prepared.append(case)

    suite_result = rag_engine.run_evaluation_suite(prepared)

    return EvalSuiteResponse(
        n_cases=suite_result["aggregate"]["n_cases"],
        n_evaluated=suite_result["aggregate"]["n_evaluated"],
        n_gold_matched=suite_result["aggregate"].get("n_gold_matched", 0),
        rag_means=suite_result["aggregate"]["rag_means"],
        baseline_means=suite_result["aggregate"]["baseline_means"],
        kappa_means=suite_result["aggregate"].get("kappa_means", {}),
        hallucination_f1=suite_result["aggregate"].get("hallucination_f1", {}),
        wins=suite_result["aggregate"]["wins"],
        results=suite_result["results"],
        disclaimer=DISCLAIMER,
    )


class GoldCaseOut(BaseModel):
    text: str
    correct_helplines: List[str]
    correct_sections: List[str]
    expected_domains: List[str]
    difficulty: str


class GoldDatasetResponse(BaseModel):
    n_cases: int
    cases: List[GoldCaseOut]


@app.get("/api/gold-dataset", response_model=GoldDatasetResponse)
def gold_dataset_api() -> GoldDatasetResponse:
   
    return GoldDatasetResponse(
        n_cases=len(rag_engine.GOLD_DATASET),
        cases=[
            GoldCaseOut(
                text=g.text,
                correct_helplines=sorted(g.correct_helplines),
                correct_sections=sorted(g.correct_sections),
                expected_domains=g.expected_domains,
                difficulty=g.difficulty,
            )
            for g in rag_engine.GOLD_DATASET
        ],
    )


@app.exception_handler(Exception)
def generic_exception_handler(_request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": str(exc),
            "disclaimer": DISCLAIMER,
        },
    )

