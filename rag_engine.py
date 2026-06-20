from __future__ import annotations
import hashlib
import json
import os
import random
import re
import statistics
import time
from google import genai
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    print("[rag_engine] ChromaDB / sentence-transformers not installed — "
          "falling back to TF-IDF retrieval.")

try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import anthropic as _anthropic_module
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


@dataclass
class KBEntry:
    id: str
    domain: str
    title: str
    content: str


KNOWLEDGE_BASE: List[KBEntry] = [
    KBEntry('fr_law', 'fundamental_rights', 'Constitutional rights overview', 'Article 14 guarantees equality before law and equal protection. Article 15 prohibits State discrimination on grounds of religion, race, caste, sex or place of birth. Article 16 guarantees equal opportunity in public employment. Article 19 protects freedoms such as speech, expression, assembly and movement. Article 21 protects life and personal liberty — interpreted to include dignity, privacy, livelihood and health. Article 22 protects against illegal arrest: you must be told why you are arrested, you have the right to a lawyer, and you must be produced before a magistrate within 24 hours. These rights are enforceable mainly against the State or public authorities, not always against private individuals.'),
    KBEntry('fr_action', 'fundamental_rights', 'How to act on a Fundamental Rights violation', 'If a Fundamental Right is violated by the State or a public authority: (1) Collect evidence — orders, notices, letters, photos, or witness statements. (2) File a complaint with the National Human Rights Commission (nhrc.nic.in) or the State Human Rights Commission. (3) For urgent violations of life and liberty, file a Writ Petition in the High Court (Article 226) or Supreme Court (Article 32). (4) For discrimination in government employment, approach the Central Administrative Tribunal (CAT). Key helplines: NHRC 14433; Women Helpline 181; SC/ST helpline 14566. You do not need a lawyer to file an NHRC complaint — it can be done online at nhrc.nic.in.'),
    KBEntry('fr_police_illegal', 'fundamental_rights', 'Illegal arrest and police detention rights', 'Under Article 22 of the Constitution and the BNSS 2023: You CANNOT be held for more than 24 hours without being presented to a magistrate. You MUST be told the reason for your arrest. You HAVE the right to inform a family member or friend. You HAVE the right to consult a lawyer of your choice (you can refuse to answer questions until your lawyer arrives). If police refuse to show an arrest memo or deny these rights, this is illegal. Action: File a Habeas Corpus petition in the High Court. Inform a lawyer immediately. Contact the State Legal Services Authority (SLSA) for free legal aid — every district has one. Helpline for legal aid: NALSA 15100 (free, 24x7).'),
    KBEntry('fr_discrimination_caste', 'fundamental_rights', 'Caste discrimination and SC/ST protection', 'The Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989 (amended 2016) punishes anyone who: humiliates an SC/ST person publicly, forces them to do degrading work, denies them access to water sources or land, or commits violence against them because of their caste. This is a COGNIZABLE and NON-BAILABLE offence — police must register an FIR and investigate. Action: File an FIR at the nearest police station (Zero FIR if needed). If police refuse, complain to the Superintendent of Police or the District Magistrate. SC/ST helpline: 14566. Special courts (Exclusive Special Courts) hear these cases with faster timelines.'),
    KBEntry('fr_privacy', 'fundamental_rights', 'Right to Privacy', "The Supreme Court (K.S. Puttaswamy v. Union of India, 2017) confirmed that privacy is a Fundamental Right under Article 21. This covers: protection of personal data, body autonomy, informational privacy, and freedom from surveillance without legal basis. If a private company, employer, or government agency misuses your personal data (Aadhaar, phone number, biometric): (1) Send a written complaint to the company's grievance officer (mandatory under most regulations). (2) File a complaint with the Data Protection Board once the Digital Personal Data Protection Act 2023 is fully notified. (3) For Aadhaar misuse, contact UIDAI at 1947 or resident.uidai.gov.in. For surveillance or wiretapping without court order, approach the High Court."),
    KBEntry('env_rights', 'fundamental_rights', 'Environmental rights and pollution complaints', 'Article 21 includes the right to a clean and healthy environment (expanded by the Supreme Court). Key laws: Environment Protection Act 1986, Water Act 1974, Air Act 1981, Noise Pollution Rules 2000. For pollution by a factory or industry: (1) File a complaint with the State Pollution Control Board (SPCB) — they have online portals. (2) For water/river pollution, also complain to the National River Conservation Directorate. (3) For noise pollution: Noise above 55 dB (day)/45 dB (night) in residential areas is illegal — complain to local police or PCB. National Green Tribunal (NGT): A special court for environmental matters. You can file an application at ngtnational.gov.in (small filing fee, no lawyer required). NGT can quickly order companies to stop polluting and pay compensation. CPCB (Central Pollution Control Board): Complaints online at envisnaturate.nic.in.'),
    KBEntry('fr_free_speech', 'fundamental_rights', 'Freedom of speech and expression limits', 'Article 19(1)(a) guarantees freedom of speech and expression. Reasonable restrictions (Article 19(2)) are permitted on grounds of sovereignty, security, public order, decency, defamation, incitement to offence. Section 153A BNS punishes promoting enmity between groups on grounds of religion, race, caste, etc. Section 196 BNS covers sedition-like offences (acts endangering sovereignty). Social media posts: can be taken down under IT Act Section 69A by government order; you can challenge such orders before the High Court. Defamation: Both civil defamation (suit for damages) and criminal defamation (BNS Section 356) are legal remedies for the aggrieved party. Truth is a complete defence in civil defamation; in criminal defamation, truth must also be in public interest. Press freedom: Journalists are protected by Article 19 but must comply with the same restrictions. Contempt of Court Act 1971 can restrict media coverage of ongoing trials.'),
    KBEntry('fr_religion', 'fundamental_rights', 'Freedom of religion rights (Articles 25–28)', 'Article 25: Every person has the right to freely profess, practise and propagate religion, subject to public order, morality and health. Article 26: Religious denominations can manage their own religious affairs. Article 27: No one can be compelled to pay taxes for promotion of any religion. Article 28: No religious instruction in State-funded institutions; optional in State-aided minority institutions. Anti-conversion laws: Several states (UP, MP, Gujarat, etc.) have enacted laws requiring prior government permission for religious conversions. Violation: report to the local magistrate. Minority rights: Linguistic and religious minorities can establish and administer their own educational institutions (Article 30). For religious discrimination by a government office, file a complaint with the NHRC (nhrc.nic.in) or relevant Human Rights Commission.'),
    KBEntry('fr_education_child', 'fundamental_rights', 'Right to Education as a Fundamental Right', 'Article 21A (inserted by 86th Amendment 2002): The State shall provide free and compulsory education to all children aged 6–14. The Right to Education Act 2009 operationalises this right. No child can be expelled or required to pass a board exam before completing elementary education (Classes 1–8). Children with disabilities: Right to inclusive education under the Rights of Persons with Disabilities Act 2016 (RPWD). Children in conflict with law: Juvenile Justice (Care and Protection of Children) Act 2015 ensures education in observation/special homes. Out-of-school child: Any person can report to the Block Education Officer or District Education Officer. NCPCR (National Commission for Protection of Child Rights) can be approached for systemic violations at ncpcr.gov.in.'),
    KBEntry('fr_equality_gender', 'fundamental_rights', 'Gender equality and anti-discrimination rights', 'Article 15(3) permits the State to make special provisions for women and children — basis for reservations and welfare laws. Article 16(2): No discrimination in public employment on grounds of sex. Transgender persons: Transgender Persons (Protection of Rights) Act 2019 prohibits discrimination in employment, education, healthcare. Right to self-perceived gender identity recognised by Supreme Court (NALSA v. Union of India, 2014). For discrimination in government employment: file a complaint with Central Administrative Tribunal (CAT) or approach the High Court. For private sector discrimination: POSH Act (sexual harassment), Equal Remuneration Act 1976 (equal pay). National Commission for Women (NCW): file complaints at ncw.nic.in or call 7827170170. For transgender rights violations: file complaints with the National Council for Transgender Persons.'),
    KBEntry('fr_disability', 'fundamental_rights', 'Rights of persons with disabilities', 'Rights of Persons with Disabilities Act 2016 (RPWD): recognises 21 types of disabilities. Guaranteed rights: equal opportunity in employment, free education up to age 18, barrier-free access to public buildings. Reservation: 4% reservation in central government jobs for PwDs (1% each: blindness/low vision; deaf/hard of hearing; locomotor disability/cerebral palsy; others). Disability certificate: issued by a government medical board — apply at your district hospital. UDID (Unique Disability ID) card: apply at swavlambancard.gov.in — gives access to government schemes. If denied access to a public building/transport: file a complaint with the Chief Commissioner for Persons with Disabilities (disabilityaffairs.gov.in). For employment discrimination: approach the Equal Opportunity Officer of the establishment, or file with the State Commissioner for Persons with Disabilities. Assistive aids (hearing aids, crutches, etc.) are provided free under ADIP scheme via district social welfare offices.'),
    KBEntry('fr_bonded_labour', 'fundamental_rights', 'Prohibition of bonded labour and trafficking', 'Article 23: Prohibits human trafficking, begar (forced labour) and other similar forms of forced labour. Bonded Labour System (Abolition) Act 1976: Any bonded labour agreement is void. All bonded labourers are deemed to have been released. Their debts are extinguished — they cannot be forced to work to repay the bonded debt. If you discover bonded labour: report to the District Magistrate immediately. The DM must conduct a survey and rescue within 24 hours. Anti-Human Trafficking Units (AHTUs): established in districts across India — contact your local police or CHILDLINE 1098. Trafficking (BNS Section 143): punishable with 7–10 years imprisonment; aggravated trafficking with 10 years to life. Victims: entitled to compensation under the Nirbhaya Fund and National Anti-Trafficking Relief and Rehabilitation Framework. iGOT portal for awareness; Shakti Vahini NGO (011-47512400) assists trafficking survivors.'),
    KBEntry('fr_child_rights', 'fundamental_rights', 'Child rights and protection framework', 'Article 24: Children below 14 CANNOT be employed in factories, mines or hazardous employment. Child Labour (Prohibition and Regulation) Amendment Act 2016: total ban on employment below 14; restricted employment for 14–18 in hazardous work. POCSO Act 2012 (Protection of Children from Sexual Offences): protects children under 18 from sexual assault, harassment and pornography. Mandatory reporting: Any person who knows of a POCSO offence MUST report to police or CHILDLINE 1098 — failing to report is itself an offence. Juvenile Justice Act 2015: Children in need of care (CNCP) are produced before Child Welfare Committee (CWC), not police. NCPCR: National Commission for Protection of Child Rights — file complaints at ncpcr.gov.in or call 1800-11-5545. State Commissions for Protection of Child Rights (SCPCRs): for state-level complaints. CHILDLINE 1098: free, 24x7, connects to nearest CHILDLINE centre for rescue, rehabilitation and legal support.'),
    KBEntry('fr_internet_access', 'fundamental_rights', 'Right to internet access and digital rights', "The Kerala High Court (2019) and the Supreme Court (Anuradha Bhasin v. Union of India, 2020) have held that the right to internet access is part of freedom of speech (Article 19) and right to livelihood (Article 21). Internet shutdowns: Must be ordered under the Temporary Suspension of Telecom Services (Public Emergency) Rules 2017. Orders must be reasoned, time-limited and reviewed by a Review Committee. Indefinite shutdowns are illegal — challenge by filing a Writ Petition in the High Court or Supreme Court. Social media accounts wrongly blocked: Apply to the platform's appeal/review mechanism; if systemic, approach the IT Grievance Appellate Committee under IT (Intermediary Guidelines) Rules 2021. Grievance Appellate Committee: appeal.goionline.in — for grievances against platforms like Twitter/X, Facebook, Instagram, WhatsApp. ISP blocking website: File a complaint with TRAI (Telecom Regulatory Authority of India) at trai.gov.in."),
    KBEntry('fr_food_security', 'fundamental_rights', 'Right to food and social security', "The Supreme Court has held the right to food to be part of the right to life under Article 21. National Food Security Act 2013: 75% of rural and 50% of urban population are entitled to subsidised foodgrains (5 kg/person/month at ■1–3/kg) through the Public Distribution System (PDS). Priority Households and Antyodaya families covered under ration card system. If ration card is denied/delayed: Apply at the district food and civil supplies office; file RTI for status; escalate to State Food Commission. Mid-Day Meal Scheme: Government school children (Classes 1–8) are entitled to free cooked meals on school days. PMGKAY (Pradhan Mantri Garib Kalyan Anna Yojana): additional free foodgrain during emergencies — check portal for current status. For denial of PDS entitlements: file a complaint at the National Food Security portal or the state government's food department grievance portal. ICDS (Integrated Child Development Services): nutrition, health and pre-school education for children under 6 — via Anganwadi centres."),
    KBEntry('fr_vote', 'fundamental_rights', 'Right to vote and electoral rights', 'Every citizen aged 18+ who is not disqualified has the right to vote (Article 326). Voter registration: enroll or update at voterportal.eci.gov.in or via Form 6 at your nearest ERO (Electoral Registration Officer). Voter ID / EPIC card: apply at voterportal.eci.gov.in or Voter Helpline 1950. Model Code of Conduct (MCC): enforced during elections — candidates/parties cannot make cash/gift inducements. Report MCC violations to the ECI. cVIGIL App: citizens can report electoral violations (bribery, booth capture, misuse of government resources) directly to the ECI in real time. NOTA (None of the Above): available on EVMs as an option. Postal ballot: available for senior citizens (80+), PwDs, essential service workers and NRIs (with conditions). Right to campaign: political canvassing is protected under Article 19 as long as it does not violate MCC or law. If denied ballot/removed from rolls: approach the Returning Officer; file a complaint with the District Election Officer.'),
    KBEntry('fr_housing', 'fundamental_rights', 'Right to shelter and housing', "The Supreme Court has interpreted the right to life (Article 21) to include the right to shelter and livelihood. PM Awas Yojana (Urban and Gramin): central housing scheme for economically weaker sections and low-income groups. Apply at pmaymis.gov.in (urban) or pmayg.nic.in (rural). Forced eviction (demolitions by government): Must follow due process — notice, hearing, alternative accommodation. Arbitrary 'bulldozer action' without legal process is unconstitutional — the Supreme Court has repeatedly held this. If facing eviction: file a Writ Petition in the High Court seeking stay. Contact NALSA (15100) for free legal aid. Slum rehabilitation: State Slum Rehabilitation Authorities handle resettlement — residents must be consulted. Pradhan Mantri Gramin Awaas Yojana: provides ■1.20–1.30 lakh to BPL rural households for pucca houses. Night shelters: Urban Local Bodies are required to provide night shelters for homeless persons — complain to the local District Social Welfare Officer if unavailable."),
    KBEntry('fr_minority', 'fundamental_rights', 'Rights of linguistic and religious minorities', 'Article 29: Any section of citizens with a distinct language, script or culture has the right to conserve it. Article 30: Minorities (religious or linguistic) have the right to establish and administer educational institutions. State cannot discriminate against minority educational institutions in granting aid. National Commission for Minorities (NCM): file complaints at ncm.nic.in or call 011-23382353 for violations against religious minorities (Muslims, Christians, Sikhs, Buddhists, Zoroastrians, Jains). National Commission for Linguistic Minorities: file at nclm.nic.in. For violence against minorities: file FIR under relevant BNS sections + Section 153A (promoting enmity). Waqf properties (Muslim religious/charitable): disputes handled by State Waqf Boards and Waqf Tribunals. Endowment disputes (Hindu religious institutions): handled by State Endowment/HR&CE; Departments. Tribal rights: Scheduled Tribes and Other Traditional Forest Dwellers (Recognition of Forest Rights) Act 2006 grants tribal communities rights to forest land.'),
    KBEntry('fr_fair_trial', 'fundamental_rights', 'Right to fair trial and legal aid', 'Article 21 includes the right to a fair and speedy trial. Article 22(1): You cannot be denied the right to consult and be defended by a legal practitioner of your choice. Free legal aid: Article 39A mandates the State to ensure equal justice. The Legal Services Authorities Act 1987 provides free legal services to: persons earning less than ■1 lakh/year, women, children, SC/ST persons, disabled persons, persons in custody, victims of natural disaster. NALSA (National Legal Services Authority): helpline 15100 — connects to free lawyers in every district. District Legal Services Authority (DLSA): walk in to your district court complex. Lok Adalats: Alternative dispute resolution — settlements here are final, binding and NOT subject to appeal. Good for motor accident claims, matrimonial disputes, bank recovery cases. Fast Track Courts: for POCSO, rape, acid attack cases. The right to speedy trial: if a trial is unduly delayed, approach the High Court by filing a petition seeking expedited hearing.'),
    KBEntry('fr_rti_info', 'fundamental_rights', 'Right to information as a fundamental right', 'The Supreme Court (Union of India v. Association for Democratic Reforms, 2002) held that the right to know is part of freedom of speech under Article 19(1)(a). RTI Act 2005 codifies this — any citizen can seek information from public authorities within 30 days (48 hours for life/liberty). Electoral bonds, political funding: Supreme Court (2024) declared Electoral Bonds scheme unconstitutional — political parties must disclose donors. Right to know criminal antecedents of electoral candidates: enshrined by Supreme Court. Restrictions on RTI: Section 8 exempts ten categories including national security, Cabinet proceedings, personal information. RTI for private information about another person: permissible if larger public interest is served. RTI for judicial orders/judgments: courts are subject to RTI except for deliberative processes. Section 7(1) BNSS: accused has the right to access documents on which the prosecution relies before trial. Whistleblower protection: Whistle Blowers Protection Act 2014 — file at Central Vigilance Commission.'),
    KBEntry('fr_humane_prison', 'fundamental_rights', 'Rights of prisoners and undertrial detainees', 'Article 21 applies even after conviction — no one loses their fundamental rights merely because they are incarcerated. Prisoners have the right to: medical treatment, adequate food and water, protection from torture, legal aid. Undertrial prisoners: Section 479 BNSS — if an undertrial has spent half the maximum period of punishment in custody, they are entitled to bail (some exceptions). Zero FIR rule and remand: police MUST produce an arrested person before a magistrate within 24 hours. If relatives are not informed of arrest: Right to inform a family member under BNSS — violation should be reported to the Magistrate. Bail: Personal Recognizance bail (PR bond) granted by Magistrate for bailable offences — accused need not pay money bail if they cannot afford it. Suomoto cognizance: Courts can take cognizance of prison condition reports. File complaints about prison conditions to: National Human Rights Commission (NHRC) at nhrc.nic.in or the State Human Rights Commission. Criminal Law & Public Safety 4 existing | 16 new | 20 total'),
    KBEntry('cl_law', 'criminal_law', 'Criminal law framework (BNS / BNSS / BSA 2023)', 'The Bharatiya Nyaya Sanhita (BNS) 2023 replaced the IPC. Key sections: Section 109–110: attempt to murder/culpable homicide. Section 115: voluntary causing hurt/grievous hurt. Section 126: wrongful restraint/confinement. Section 132: criminal force and assault. Section 135–136: kidnapping and abduction. Section 296: criminal intimidation (threats). Section 316–317: cheating (fraud). Section 303: theft. Section 308: extortion. Sexual offences (Chapter V): rape, sexual assault, stalking, voyeurism — with enhanced penalties. The Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 replaced the CrPC — sets out FIR, investigation, arrest, bail procedures. The Bharatiya Sakshya Adhiniyam (BSA) 2023 governs evidence including electronic evidence.'),
    KBEntry('cl_action', 'criminal_law', 'How to file an FIR and follow up', "Step 1: Go to the nearest police station and give a written complaint (or verbal — they must write it down). Step 2: A 'Zero FIR' can be filed at ANY station regardless of jurisdiction — it will be transferred. Step 3: Police MUST give you a free copy of the FIR. The FIR number is your tracking reference. Step 4: If police REFUSE to register an FIR for a cognizable offence — (a) Write to the Superintendent of Police by registered post. (b) File a complaint before the Executive Magistrate (Section 175 BNSS). (c) Send a complaint to the State's Director General of Police online. Step 5: Track the case using the FIR number on your State's police portal. National Emergency: 112. Women safety: 1091. Child helpline: 1098."),
    KBEntry('cl_women_safety', 'criminal_law', "Sexual harassment and women's safety laws", 'Key legal protections: (1) POSH Act 2013: covers sexual harassment at the workplace — every employer with 10+ employees MUST have an Internal Complaints Committee (ICC). You can file a complaint with the ICC within 3 months of the incident. (2) BNS Section 74–78: punishes eve-teasing, stalking, voyeurism, sexual assault. Stalking (repeatedly following, messaging, monitoring online) is a criminal offence. (3) Protection of Women from Domestic Violence Act 2005: covers physical, verbal, emotional and economic abuse. (4) Acid attack: BNS Section 124 — severe punishment; free treatment in government hospitals is mandatory. Immediate action: Call 112 (emergency) or 1091 (women helpline). File a Zero FIR at the nearest police station. Preserve all evidence: screenshots, messages, photos of injuries, witness names. One Stop Centres (Sakhi Centres) in every district provide shelter, legal aid, and counselling — free of cost.'),
    KBEntry('cl_threat_extortion', 'criminal_law', 'Threats and extortion', 'Receiving threats (BNS Section 296) and extortion (BNS Section 308) are cognizable criminal offences. Extortion means someone threatens you to force you to give money, property or any advantage. Evidence to preserve: screenshots of WhatsApp/SMS/email threats, call recordings (check state law on consent), witness names, dates/times of incidents. Action steps: (1) File an FIR immediately at the nearest police station. (2) For online threats, ALSO report at cybercrime.gov.in or call 1930. (3) Apply to the Magistrate for an anticipatory bail (if you fear arrest) or protection order. (4) If the person is known to you, you can also seek a restraining order from a civil court. Do NOT pay — paying encourages more demands and is itself evidence of the threat.'),
    KBEntry('cl_domestic_cruelty', 'criminal_law', 'Cruelty by husband and in-laws (BNS Section 86)', 'BNS Section 86 (replacing IPC Section 498A): Husband or relative of husband subjecting a woman to cruelty is punishable with up to 3 years imprisonment and fine. Cruelty includes: wilful conduct causing grave injury to life/limb/health, or harassment to coerce the woman or her relatives to meet unlawful dowry demands. This is a cognizable and non-bailable offence — police must register an FIR. Action: File an FIR at the nearest police station; also file an application under the Domestic Violence Act before the Magistrate for a protection order. One Stop Centres (Sakhi Centres) in every district provide free shelter, legal aid, medical and counselling support. If police refuse to register FIR: write to the SP or approach the Executive Magistrate. Anticipatory bail: In-laws sometimes file counter-cases — consult a lawyer immediately if threatened with arrest. Legal aid: NALSA 15100 — free legal representation for women in these cases.'),
    KBEntry('cl_dowry_death', 'criminal_law', 'Dowry death and dowry demand laws', "Dowry Prohibition Act 1961: Giving or taking dowry is a criminal offence — punishable with minimum 5 years imprisonment + fine. BNS Section 80 (Dowry death): If a woman dies within 7 years of marriage due to burns, bodily injury or under suspicious circumstances, and it is shown she was subjected to cruelty or dowry harassment, it is presumed to be a dowry death. Punishment: Not less than 7 years, may extend to life. Action: File an FIR immediately — this is a cognizable, non-bailable, non-compoundable offence. Demand for dowry: Even a verbal demand is an offence under the Dowry Prohibition Act — file a complaint with the Dowry Prohibition Officer (appointed in each district). Evidence to collect: letters, WhatsApp messages, witness statements, receipts of gifts given. NHRC: If police are inactive, complain to the National Human Rights Commission. Victim's parents: Can also file a civil suit to recover dowry articles."),
    KBEntry('cl_acid_attack', 'criminal_law', 'Acid attack laws and victim rights', 'BNS Section 124: Throwing or administering acid on any person causing permanent/partial damage is punishable with minimum 10 years (may extend to life) + fine. Acid sale is regulated: sellers must maintain a register of buyers; sale to anyone below 18 is prohibited. Victim rights (BNS Section 397): Victims of acid attacks are entitled to free medical treatment at ALL government hospitals, and also at private hospitals initially. Compensation: Minimum ■3 lakh to be paid by the State to acid attack victims under Nirbhaya Fund guidelines. Courts often award additional compensation from the accused. Immediate action: Get to the nearest hospital IMMEDIATELY; do not apply anything to the wound before professional treatment. Call 112. File an FIR at the police station. Rehabilitation: Chhanv Foundation (www.chhanv.org, 011-33106860) and Make Love Not Scars NGO provide medical, legal and livelihood support to survivors.'),
    KBEntry('cl_cyberstalking', 'criminal_law', 'Stalking and cyberstalking laws', "BNS Section 78: Stalking — following, contacting, or monitoring a woman against her clearly expressed disinterest is a criminal offence. First conviction: 3 years imprisonment + fine. Second conviction: 5 years + fine. Cyberstalking: monitoring someone's online activities, sending repeated messages, tracking location via apps — all covered. Action: (1) Collect screenshots of all messages, calls, social media contacts. (2) File an FIR at the nearest police station. (3) Also file at cybercrime.gov.in. If the stalker is known: apply to a Magistrate for a restraining order / protection order under CrPC/BNSS. If the stalker is a former partner: also file under the Domestic Violence Act for a protection order. National Commission for Women: complaint at ncwapps.nic.in or call 7827170170. Safety tip: Change all passwords and security settings; remove unknown devices from your accounts."),
    KBEntry('cl_cheating_fraud', 'criminal_law', 'Cheating and fraud offences (BNS)', 'BNS Section 316–318 (Cheating): Dishonestly inducing a person to deliver property or alter their legal position is cheating — punishable with up to 7 years + fine. Section 319 (Cheating by impersonation): Personating another person to deceive — up to 5 years + fine. Section 338–339 (Criminal breach of trust): Entrusted with property and dishonestly misappropriating it — up to 7 years + fine for aggravated cases. Common scenarios: advance fee fraud, job fraud (took money/documents and disappeared), matrimonial fraud, investment scams. Action: (1) File an FIR. (2) For online fraud additionally report at cybercrime.gov.in and call 1930. (3) For investment fraud (Ponzi scheme), report to SEBI at scores.sebi.gov.in and to the state EOW (Economic Offences Wing). Evidence: bank transaction receipts, WhatsApp messages, email correspondence, job offer letters, receipts of money paid. Attach money attachment: if fraudster is traceable, apply to Magistrate under BNSS for provisional attachment of assets.'),
    KBEntry('cl_murder_culpable', 'criminal_law', 'Murder, culpable homicide and self-defence', 'BNS Section 100–101: Murder is unlawful killing with intention to cause death or bodily injury likely to cause death. Punishment: Death or life imprisonment. BNS Section 105: Culpable homicide not amounting to murder — less severe intent — punishable with up to 10 years or life. Self-defence (Right of Private Defence, BNS Section 34–44): You have the right to defend yourself and others using proportional force. You can use force that may cause death if you have reasonable cause to apprehend death or grievous hurt to yourself. The right does not extend to taking vengeance after the threat has passed. If someone in your family is killed: File an FIR immediately. If the accused are powerful, also send a complaint to the SP and the DIG. Witness protection: Under the Witness Protection Scheme 2018, witnesses in serious cases can get police protection — apply to the trial court. Victim compensation: File an application in the trial court under the Victims Compensation Scheme of your State.'),
    KBEntry('cl_kidnapping_abduction', 'criminal_law', 'Kidnapping and abduction laws', 'BNS Section 137: Kidnapping from India (taking a person out of India) or from lawful guardianship (taking a minor or unsound person) — punishable with up to 7 years. BNS Section 138: Abduction — compelling or inducing any person to go from one place — by force or deceit. BNS Section 140: Kidnapping for ransom — minimum 7 years, may extend to life or death. Child missing: Call CHILDLINE 1098 (24x7). Register on the Track Child portal (trackthemissingchild.gov.in). File an FIR immediately — police cannot refuse to register an FIR for a missing child citing 24-hour rule; this rule does NOT apply to children. Parental child abduction (one parent taking child without court order in custody dispute): File a Habeas Corpus petition in the High Court. NRI parental abduction (child taken abroad): Contact the Ministry of External Affairs and the Embassy in the destination country immediately.'),
    KBEntry('cl_sexual_assault_men', 'criminal_law', 'Sexual offences — scope, procedure and evidence', "BNS Chapter V covers sexual offences: rape (Section 63), aggravated rape (Section 65–70), sexual assault, stalking, voyeurism. Rape is gender-neutral for victims from 2023 onwards in certain provisions. Marital rape: Currently not an offence under Indian law for adults (ongoing legal debate); but physical abuse within marriage is covered by PWDVA and BNS Section 86. Two-finger test: Banned by Supreme Court — courts are instructed not to use or admit such reports. Evidence: Medical examination should be done as soon as possible; DNA evidence from Forensic Science Laboratory (FSL). Electronic evidence: Screenshots, location data, chat history — certified under BSA 2023. Survivors' identity: Cannot be disclosed by media or anyone — violation is a criminal offence (BNS Section 72). In-camera trial: Sexual offence cases are tried in camera (closed court) to protect the survivor's privacy. Free legal aid: Every DLSA must provide a lawyer to survivors of sexual offences — NALSA 15100."),
    KBEntry('cl_mob_lynching', 'criminal_law', 'Mob violence, lynching and hate crimes', "BNS Section 103(2): Murder committed by a group of five or more persons acting in concert on grounds of race, caste, community, sex, place of birth, language, or personal belief — punishable with death or life imprisonment. This provision specifically addresses mob lynching and communal violence. If you witness or are a victim of mob violence: Call 112 immediately. File an FIR — the specific section ensures life or death punishment for all members of the mob. If police are inactive or complicit: File a complaint with the State Human Rights Commission (SHRC) or NHRC at nhrc.nic.in. File a public interest petition or writ before the High Court. Compensation: Apply under the State's Victim Compensation Scheme or approach the SHRC for ex-gratia. Rights of accused in mob violence: All arrested persons still have constitutional rights under Article 22."),
    KBEntry('cl_bail_process', 'criminal_law', 'Bail rights and application process', "Types of bail in India: Regular bail (Section 480 BNSS): for offences already arrested for. Anticipatory bail (Section 482 BNSS): before arrest, if apprehending arrest. Bail on default (Section 479 BNSS): if trial is not concluded within the prescribed period. Bailable offences: Accused has a right to bail — police or magistrate must grant it. Non-bailable offences: Bail is at the court's discretion; magistrate considers: nature of offence, likelihood of tampering with evidence, flight risk. NDPS Act, UAPA, PMLA: Special provisions — bail is harder to get; prosecution must be heard; strict conditions. Bail conditions: Surrender passport, report to police station periodically, do not contact witnesses. Bail application: Can be filed by the accused or their lawyer in the Magistrate's Court, Sessions Court or High Court (for higher offences). Free legal aid for bail: NALSA 15100 — free lawyer for undertrial prisoners."),
    KBEntry('cl_witness_victim', 'criminal_law', 'Rights of victims and witnesses in criminal cases', "Victim's rights under BNSS 2023: Right to be informed of progress of investigation. Right to be heard before bail is granted to the accused in heinous offences. Right to a copy of the charge sheet. Right to engage a private lawyer to assist the prosecution. Victim compensation: Section 396 BNSS — courts can award compensation from the accused. Also apply under the State Victim Compensation Scheme (all states mandated by Supreme Court). Witness protection: Witness Protection Scheme 2018 — apply to the trial court for protection measures (pseudonym, screen, video deposition). Child witness: Must be examined in a child-friendly room or via video link (POCSO cases). Hostile witness: If a witness turns hostile, the prosecutor can cross-examine them with permission of the court."),
    KBEntry('cl_juvenile_crime', 'criminal_law', 'Juvenile Justice and children in conflict with law', "Juvenile Justice (Care and Protection of Children) Act 2015: Children below 18 who commit offences are treated as 'children in conflict with law'. For offences below 7 years imprisonment: handled by Juvenile Justice Board (JJB) — goal is rehabilitation, not punishment. For heinous offences (7+ years max punishment) by 16–18 year olds: JJB may transfer to Children's Court (Sessions Court) to be tried as an adult. Children cannot be sentenced to death or life imprisonment. Observation homes: Children are kept in observation homes (not prisons) during proceedings. Child's identity: Cannot be disclosed by media — strict prohibition. Rights in JJB: Child must be accompanied by parents/guardian. Free legal aid through DLSA. Rehabilitation: JJBs must formulate individual care plans — education, skill development, counselling. CHILDLINE 1098 responds to children in conflict with law and connects them to appropriate legal aid."),
    KBEntry('cl_forgery_false_doc', 'criminal_law', 'Forgery and false document offences', "BNS Section 336–337: Forgery — making a false document or false electronic record with intent to cause damage or injury to any person or defraud — punishable with up to 2 years (simple) or up to 7 years (for specific documents like court records, government records, valuable securities). BNS Section 340: Using as genuine a forged document — punishable equally to the forger. BSA 2023 Section 22: Electronic records are admissible as evidence when certified by the responsible official. Common scenarios: forged property documents, fake degree/caste certificates, forged signatures on cheques/loan documents. Action: (1) File an FIR with original documents as evidence. (2) For forged property documents: also file complaint with the Sub-Registrar's office. (3) For fake certificates: complain to the issuing authority and the verification portal. Cross-examination of documents: in civil/criminal cases, documents can be disputed through handwriting experts from the FSL."),
    KBEntry('cl_drug_ndps', 'criminal_law', 'NDPS Act and drug-related offences', 'Narcotic Drugs and Psychotropic Substances Act 1985 (NDPS): governs offences relating to drug possession, use, production, sale and trafficking. Small quantity possession (e.g., <100g cannabis): rigorous imprisonment up to 1 year or fine up to ■10,000 or both. Intermediate quantity: up to 10 years + fine. Commercial quantity: minimum 10 years, maximum 20 years + fine of ■1–2 lakh. Repeat offences: sentences doubled; death penalty for repeat commercial trafficking. Bail under NDPS: Very difficult — Section 37 NDPS requires the court to be satisfied that the accused is not guilty and will not commit the offence again. If accused: insist on lawyer presence during questioning. Do not sign any statement without reading carefully. Search and seizure: Officers must follow prescribed procedure; illegal searches can be challenged. Addiction and treatment: Do not conflate addiction with criminality — courts are increasingly favouring rehabilitation for small-quantity cases.'),
    KBEntry('cl_money_laundering', 'criminal_law', 'Money laundering, PMLA and financial crimes', "Prevention of Money Laundering Act 2002 (PMLA): Laundering proceeds of a scheduled offence is punishable with 3–7 years; certain offences carry up to 10 years. Enforcement Directorate (ED): investigates PMLA — has powers of arrest, attachment of property, search and seizure. Property attachment: ED can provisionally attach assets without court order — but must confirm attachment within 60 days before the Adjudicating Authority. Special bail conditions under PMLA: like NDPS, very strict — similar 'twin conditions' apply. Economic Offences Wing (EOW): state police units that investigate large-scale financial frauds (investment frauds, Ponzi schemes). SEBI: Investment fraud, insider trading, market manipulation — file complaints at scores.sebi.gov.in or call 1800-22-7575. SFIO (Serious Fraud Investigation Office): investigates corporate fraud — report at sfio.nic.in. Income tax evasion: Vigilance reports can be filed at incometaxindiaefiling.gov.in."),
    KBEntry('cl_communal_violence', 'criminal_law', 'Communal violence and riot laws', 'BNS Sections 191–195: Rioting — five or more persons with a common unlawful object using force or violence — punishable with up to 2 years (Section 191) or up to 3 years if armed (Section 195). BNS Section 153A: Promoting enmity between groups on grounds of religion, race, caste, etc. — up to 3 years + fine. In a riot, ALL members of an unlawful assembly are liable for offences committed by any member in prosecution of the common object (BNS Section 190). Curfew: Magistrates can impose curfew under BNS Section 163 to prevent violence. Violation of curfew is itself an offence. If you are displaced by communal violence: (1) Contact the District Magistrate for relief/rehabilitation. (2) Register as a displaced person with the police. NSA (National Security Act 1980): Preventive detention — up to 12 months; challenge by Habeas Corpus petition in High Court. Communal violence victims: apply under State Victim Compensation Scheme; approach State Human Rights Commission. Cybercrime & Information Technology 4 existing | 16 new | 20 total'),
    KBEntry('cy_law', 'cyber_it', 'Information Technology Act and cyber offences', 'IT Act 2000 (amended 2008) key sections: Section 43: unauthorized access/damage to computer system — civil remedy + compensation. Section 66: computer-related offences (hacking) — up to 3 years imprisonment. Section 66C: identity theft using electronic means — up to 3 years + fine. Section 66D: cheating by personation using computer (fake profiles, impersonation) — up to 3 years. Section 66E: privacy violation (publishing private images without consent) — up to 3 years. Section 67: publishing obscene material — up to 5 years. Section 67A: sexually explicit material — up to 7 years. Section 67B: child pornography — up to 7 years. Electronic records, screenshots, and digital logs are valid evidence under the BSA 2023 when properly certified.'),
    KBEntry('cy_action', 'cyber_it', 'How to report cyber crime step by step', "Step 1 — Report IMMEDIATELY (time matters for financial fraud): Call Cyber Crime Helpline 1930 (24x7) or visit cybercrime.gov.in. For UPI/bank fraud: ALSO call your bank helpline and block transactions. Step 2 — Preserve all evidence BEFORE doing anything else: Take screenshots of messages, transaction IDs/UTR numbers, profiles, URLs. Note down: date, time, amount, sender/receiver details. Do NOT delete chats. Step 3 — File an online complaint at cybercrime.gov.in (available 24x7, no need to visit a police station). Step 4 — For financial fraud, ask your bank to raise a 'chargeback' or 'dispute transaction'. Under RBI rules, if you report within 3 working days and it was not your fault (no OTP shared), liability may be ZERO. Step 5 — If unresolved, file a complaint with the Adjudicating Officer appointed under the IT Act in your state."),
    KBEntry('cy_upi_fraud', 'cyber_it', 'UPI and payment fraud: what to do', "Common UPI scams: fake QR codes, collect-money requests (never scan to receive money), OTP phishing calls (fake bank/TRAI/police), fake job/loan offers, remote access apps (AnyDesk, TeamViewer). CRITICAL RULE: You NEVER need to enter your UPI PIN to RECEIVE money. If someone asks for your PIN/OTP to 'complete a transfer to you' — it's a scam. Immediate steps if money is lost: (1) Call 1930 within minutes — provide UPI transaction ID. A hold can sometimes be placed before money is withdrawn. (2) Call your bank's helpline and report the transaction as fraudulent. Note the complaint reference number. (3) File complaint at cybercrime.gov.in — choose 'Financial Fraud'. (4) Visit the nearest police station with transaction screenshots for a formal FIR. (5) Report the fraudster's UPI ID/phone number to NPCI at npci.org.in/what-we-do/upi."),
    KBEntry('cy_social_media', 'cyber_it', 'Social media harassment and fake profiles', "Types of offences and reporting: (1) Fake profile using your photos/name: Report to the platform (Facebook, Instagram, etc.) using 'Report' → 'Fake Account'. Also file at cybercrime.gov.in → 'Report Other Cyber Crime'. (2) Morphed/obscene photos circulated without consent: This is an offence under IT Act Section 66E and 67. Do NOT re-share or screenshot from others — collect original URLs. File at cybercrime.gov.in urgently. (3) Cyberbullying/online harassment: Document with screenshots + dates. Block the person. File at cybercrime.gov.in. For serious threats: file an FIR for criminal intimidation under BNS Section 296. (4) Sextortion: Do NOT pay. Do NOT delete chats (these are evidence). Call 1930 or cybercrime.gov.in immediately. Cyber Peace Foundation helpline: 1800-200-3323. iCall (mental health support): 9152987821."),
    KBEntry('cy_phishing_email', 'cyber_it', 'Email phishing and identity theft', "Phishing: fraudulent emails/SMS that appear to be from a bank, government, or trusted company asking you to click a link and enter your credentials. Never click links in unsolicited emails. Always verify by navigating directly to the official website. If your email account is hacked: (1) Recover access using the platform's account recovery process. (2) Change all linked passwords. (3) Enable two-factor authentication (2FA). (4) Report the phishing email at reportphishing@apwg.org or to the platform's abuse team. Identity theft (IT Act Section 66C): Using another person's electronic signature, password, or other unique identification feature — up to 3 years + fine of ■1 lakh. Report identity theft: cybercrime.gov.in → 'Report Other Cyber Crime'. Credit bureau freeze: If your financial identity is stolen, contact CIBIL (cibil.com) or Experian to flag your account. File with cybercrime.gov.in and your bank; request a credit freeze if possible."),
    KBEntry('cy_ransomware', 'cyber_it', 'Ransomware and malware attacks on individuals and businesses', "Ransomware: malicious software that encrypts your files and demands payment (often in cryptocurrency) for the decryption key. DO NOT PAY THE RANSOM — payment does not guarantee recovery and encourages attackers. Immediate steps: (1) Disconnect the infected device from the internet and all networks immediately. (2) Report to CERT-In (Indian Computer Emergency Response Team) at incident@cert-in.org.in or call +91-1800-11-4949. (3) File a complaint at cybercrime.gov.in (choose 'Ransomware'). (4) File an FIR — offence under IT Act Section 66 (computer-related offence). (5) Do NOT delete any files — forensic investigation may be needed. Business impact: If customer data was leaked in the attack, the company may have obligations under IT Act Section 43A and forthcoming DPDP Act 2023 to notify affected persons. Decryption tools: Check nomoreransom.org (free decryption tools for many ransomware variants). Prevention: Keep backups offline/encrypted; keep OS and software updated; use reputable anti-virus."),
    KBEntry('cy_dark_web', 'cyber_it', 'Dark web misuse and data leak response', 'If your personal data (Aadhaar, bank details, passwords) appears on the dark web or in a known data breach: (1) Change all passwords immediately, starting with email, then banking. (2) Enable two-factor authentication everywhere. (3) Notify your bank — request card replacement if card data is compromised. (4) File a complaint at cybercrime.gov.in. (5) Report the data breach to CERT-In (cert-in.org.in) — they investigate breaches of Indian citizen data. Companies with a data breach: CERT-In mandates breach reporting within 6 hours of discovery (CERT-In Directions 2022). Failure is an offence. Aadhaar data breach: Report to UIDAI at 1947 or resident.uidai.gov.in — they can lock your Aadhaar biometrics. Check if your email is in a data breach: haveibeenpwned.com. Digital Personal Data Protection Act 2023 (once operational): Data Fiduciaries must notify affected Data Principals and the Data Protection Board of breaches.'),
    KBEntry('cy_online_job_fraud', 'cyber_it', 'Online job fraud and work-from-home scams', "Common scams: fake job offers asking for a registration/training fee; part-time jobs offering payment for liking videos/completing tasks (ends with a large 'deposit' that is stolen); fake HR companies stealing personal documents. Warning signs: Payment required before job starts; no verifiable company address; job offered without interview; too-good salary for minimal work. If you have been defrauded: (1) File immediately at cybercrime.gov.in or call 1930. (2) File an FIR at the local police station. (3) Send a complaint to the Ministry of Labour and Employment if the fake entity posed as a recruitment agency. Document recovery: If your documents (Aadhaar, PAN) were shared with the scammer: report misuse to the issuing authority and file a police complaint. Platform responsibility: Report the fake job post to the platform (LinkedIn, Naukri, Indeed) using the 'Report' button. National Career Service (NCS) portal: ncs.gov.in — government verified job listings; use this as a reliable source."),
    KBEntry('cy_iot_smart_device', 'cyber_it', 'Smart device, IoT security and surveillance', 'Spyware/stalkerware on mobile: If you suspect your phone is being monitored (new apps, battery draining, data spike): (1) Factory reset the phone after backing up contacts/photos. (2) Change all account passwords from a safe device. (3) Enable screen lock with a strong PIN. Hidden cameras in rental/hotel rooms: Illegal under IT Act Section 66E (violation of privacy). How to detect: Look for small holes in smoke detectors, clocks, charging adapters; use a torch — cameras reflect light. RF detector apps can also help. Action: Report to the property owner; file an FIR; report to cybercrime.gov.in. Smart home devices: Change default passwords on routers, cameras, smart bulbs. Unauthorised access to CCTV footage (by a neighbour/employer): IT Act Section 43 (unauthorized access). Drone surveillance over private property without consent: report to the Directorate General of Civil Aviation (DGCA).'),
    KBEntry('cy_fake_news', 'cyber_it', 'Misinformation, deepfakes and fake news', "Spreading false information that endangers public order or causes panic (e.g., false claims of a riot, fake health advisory): IT Act Section 66D and BNS Section 353 may apply. Deepfakes (AI-generated fake videos of real people in compromising situations): (1) IT Act Section 66E (publishing private images without consent) and 67A (sexually explicit material) apply. (2) File at cybercrime.gov.in immediately. (3) Report to the platform — platforms under MEITY's Intermediary Rules 2021 must remove deepfake content within 24 hours of being informed. Grievance Appellate Committee (appeal.goionline.in): if the platform does not remove content within time. Fact-checking: PIB Fact Check (pibfactcheck.in) for government-related claims; AltNews and Boom for independent fact-checking. Forwarding fake news: Even forwarding knowingly can attract legal liability. WhatsApp forward limit: capped at 5 — platform-level measure to slow viral misinformation."),
    KBEntry('cy_data_privacy', 'cyber_it', 'Data privacy rights under Indian law', "Digital Personal Data Protection Act 2023 (DPDP Act): once operational, gives individuals (Data Principals) the following rights over their personal data: (1) Right to know what data is collected and how it is used. (2) Right to correction and erasure of inaccurate or outdated data. (3) Right to withdraw consent. (4) Right to grievance redressal. (5) Right to nominate a representative to exercise rights on their behalf. Current protections (already in force): IT Act Section 43A — Companies must implement 'reasonable security practices'; failure causing wrongful loss entitles individuals to compensation. Sensitive personal data (financial, health, sexual orientation): requires explicit consent for collection and processing. Aadhaar data: governed by Aadhaar Act 2016 — cannot be used for purposes other than those specified. For current grievances: approach the company's Grievance Officer, then CERT-In, then file a consumer complaint."),
    KBEntry('cy_crypto_fraud', 'cyber_it', 'Cryptocurrency fraud and investment scams', "Cryptocurrency is not legal tender in India but trading is not illegal. Fake crypto investment platforms promising guaranteed returns are rampant. Common scams: 'Pig butchering' (building romantic relationship then convincing victim to invest); fake crypto exchanges; pump-and-dump coin schemes. If defrauded: (1) File at cybercrime.gov.in — choose 'Online Financial Fraud → Cryptocurrency'. (2) Call 1930. File an FIR. (3) Report the fraudulent platform to CERT-In. (4) For platforms registered as Virtual Digital Asset (VDA) service providers: report to FIU-IND (Financial Intelligence Unit) at fiuindia.gov.in. Evidence: wallet addresses, transaction hashes, screenshots of the platform and your communication. Recovery is very difficult — blockchain transactions are pseudonymous and often irreversible. SEBI warning: Invest only in SEBI-regulated entities for securities — crypto is not SEBI-regulated and has no investor protection framework."),
    KBEntry('cy_email_spoofing', 'cyber_it', 'Email spoofing, business email compromise (BEC)', "Business Email Compromise (BEC): fraudsters impersonate a company's CEO/CFO or a vendor and instruct employees to transfer funds to a fraudulent account. This is one of the highest-value cyber frauds in India. Verify all fund transfer instructions by calling the requester on a known number — never trust the number in the email. If your business has been defrauded by BEC: (1) Call 1930 immediately — banks can sometimes reverse transactions if contacted within minutes. (2) File a complaint at cybercrime.gov.in. (3) File an FIR — offence under IT Act Section 66D (cheating by impersonation) and BNS Section 316. (4) Notify your bank and the recipient bank to freeze the account. Email security: SPF, DKIM, DMARC records on your domain prevent spoofing — ask your IT team to implement these. For personal email spoofing (fake emails appearing to be from family/boss): same reporting process."),
    KBEntry('cy_online_gaming', 'cyber_it', 'Online gaming fraud and cyber offences in gaming', 'Online gaming fraud types: (1) In-game item scams — player convinces you to trade valuable items then disappears. (2) Account takeover — phishing for gaming account credentials. (3) Fake gaming platforms that collect entry fee and never pay out. (4) Cheating software (aimbots) that violates game terms and can lead to legal consequences. Online rummy/fantasy sports: legal in skill-game states; sports betting is illegal in most Indian states. For money-based gaming fraud: file at cybercrime.gov.in. Cyberbullying in games (abusive messages, threats in chat): Screenshot and report to the platform. If threats are serious, file an FIR under BNS Section 296. Loot boxes and microtransactions: the Supreme Court has been examining whether certain mechanics amount to gambling. Parents: children under 18 must not be allowed to spend real money on in-game purchases — file complaint if platform enables this without parental consent.'),
    KBEntry('cy_vpn_legal', 'cyber_it', 'VPN, encryption and legal usage in India', 'VPN (Virtual Private Network) use by individuals is currently legal in India, but: CERT-In Directions 2022 require VPN service providers operating in India to collect and store user logs (name, IP address, usage) for 5 years. Providers who do not comply must shut down India-based servers. Tor browser use: legal for legitimate privacy purposes; using it to access dark web illegal content is an offence. End-to-end encrypted messaging (WhatsApp, Signal): legal. Encryption of personal devices: legal and recommended. Government may request decryption: Under IT Act Section 69, government can direct decryption — non-compliance is an offence. What you can legally do with a VPN: Access region-locked streaming content; secure public Wi-Fi use; remote work; privacy protection. What remains illegal even with a VPN: Accessing CSAM, terrorist content, gambling sites banned in your state, copyrighted content via piracy sites.'),
    KBEntry('cy_fintech_fraud', 'cyber_it', 'Fintech app fraud and BNPL scams', "Buy Now Pay Later (BNPL) and instant loan app scams: (1) Fake loan apps that charge high processing fees upfront and disburse nothing, or disburse a small amount and demand large repayment. (2) Apps that access your contacts and photos and threaten to send morphed photos to your contacts if you don't repay. This is extortion — do NOT pay. File an FIR immediately. RBI action: RBI maintains a whitelist of approved NBFCs/fintech lenders. Only borrow from RBI-registered lenders — check rbi.org.in. If harassed by illegal recovery agents: Report to RBI at sachet.rbi.org.in and cybercrime.gov.in. KYC fraud: Never share your Aadhaar OTP or V-CIP video with unknown callers claiming to complete KYC. NACH mandate fraud: fake debit mandates created without consent — report to your bank and the NPCI at npci.org.in. For legitimate BNPL disputes: Contact the lender's grievance officer; if unresolved, approach the RBI Ombudsman."),
    KBEntry('cy_telecom_fraud', 'cyber_it', 'Telecom fraud, SIM swap and number porting scams', "SIM swap fraud: Fraudsters convince your mobile operator to transfer your number to a new SIM they control, then use your number to receive OTPs and access your bank account. Warning sign: Your SIM suddenly shows 'No Service' for an unexpected period. Immediate action: Call your telecom operator's customer care immediately to block the SIM swap. Call your bank to block transactions. Report to cybercrime.gov.in and file an FIR. TRAI regulations: Telecom operators must verify identity rigorously before SIM swap — failure is negligence. Spam calls (KYC update, TRAI blocking your number, fake police calls): These are ALL fraud. TRAI does NOT call individuals. Report spam calls: DND (Do Not Disturb) at trai.gov.in or via your telecom operator's DND portal. SANCHAR SAATHI portal (sancharsaathi.gov.in): report fraud calls, block stolen phones (CEIR), verify devices. Number porting scam: Scammers port your number to another operator. Alert your operator immediately and request a porting block."),
    KBEntry('cy_ai_misuse', 'cyber_it', 'AI-generated content misuse and legal framework', 'AI-generated fake audio (voice cloning) or video (deepfake) used to defraud or harass is illegal under existing IT Act provisions: Section 66C (identity theft), Section 66D (cheating by impersonation), Section 67A (sexually explicit material). Deepfake of public figures for political disinformation: BNS Section 353 (statements causing public mischief) + Election Commission complaint. AI-generated CSAM (child sexual abuse material): Zero tolerance — Section 67B IT Act; POCSO Act. Using AI to generate fake reviews/ratings: Consumer Protection (E-Commerce) Rules 2020 prohibit fake reviews. Report to National Consumer Helpline 1915. AI-assisted exam cheating: UGC/institution-specific regulations; may attract criminal cheating provisions under BNS. For AI-generated defamatory content about you: (1) Request takedown from the platform. (2) File at cybercrime.gov.in. (3) Pursue civil defamation remedy in court. Draft framework: Ministry of Electronics and IT (MeitY) is developing AI governance guidelines — check meity.gov.in for updates.'),
    KBEntry('cy_piracy_copyright', 'cyber_it', 'Online piracy and copyright infringement', 'Copyright Act 1957 protects original literary, dramatic, musical, artistic works and films for the lifetime of the author + 60 years. Distributing or downloading pirated content (movies, software, books) is an infringement — civil liability for damages AND criminal offence (up to 3 years + fine). Streaming pirated content on websites: also infringement (though enforcement against individual viewers is rare). Platforms like Telegram channels sharing pirated content: (1) Rights holders can send a DMCA-style takedown notice to the platform. (2) File a complaint with CERT-In or approach the court for a John Doe order (site blocking order). If you receive a copyright infringement notice: (1) Stop the infringing activity immediately. (2) If genuinely infringing, settle with the rights holder — damages can be substantial. (3) If you believe your use is legitimate (educational, criticism, parody): consult a lawyer on the fair dealing exception under Section 52 Copyright Act. Software piracy: Companies can conduct software audits and sue — purchase legitimate licences.'),
    KBEntry('cy_remote_access_scam', 'cyber_it', 'Remote access scams (AnyDesk, TeamViewer fraud)', "Remote access scam: Fraudsters call posing as bank/Microsoft/TRAI/police technical support and convince you to install AnyDesk, TeamViewer, QuickSupport or similar apps. Once you grant them access, they can see your screen, access your banking app, and steal money — all in real time. CRITICAL RULE: No legitimate bank, government or tech company will ever ask you to install a remote access app or share a remote code to 'fix' your account. If you have already given access: (1) Disconnect internet immediately — switch off Wi-Fi and mobile data. (2) Uninstall the remote access app. (3) Call your bank's helpline immediately to block transactions. (4) Change all passwords from a different, safe device. (5) Call 1930 and file at cybercrime.gov.in. The golden hour: Reporting within minutes can enable banks to place a hold on fraudulent transfers. File an FIR — offence under IT Act Section 43 (unauthorized access) and BNS Section 316 (cheating). Consumer Protection 4 existing | 16 new | 20 total"),
    KBEntry('co_law', 'consumer', 'Consumer Protection Act 2019 — your rights', "The Consumer Protection Act 2019 gives every consumer six rights: (1) Right to safety from hazardous goods/services. (2) Right to information about quality, quantity, price, standards. (3) Right to choose from a variety of products at competitive prices. (4) Right to be heard and have grievances considered. (5) Right to seek redressal — replacement, refund, or compensation for defective goods/poor service. (6) Right to consumer education. Key definitions: 'Deficiency' means any fault, imperfection, inadequacy in quality/nature/manner of performance. 'Unfair trade practice' includes misleading advertisements, false warranty, withholding genuine product. You can claim: refund + compensation + legal costs through Consumer Commissions (no court fee for small claims). Time limit: file within 2 years of the cause of action."),
    KBEntry('co_action', 'consumer', 'How to file a consumer complaint (step by step)', 'Step 1 — Send a written complaint to the company/seller first. Give them 15–30 days to resolve. Keep a copy. Step 2 — If unresolved, file online at e-daakhil.nic.in (no lawyer needed for District Commission). Or call National Consumer Helpline: 1800-11-4000 (toll-free) or 1915. Step 3 — Which Commission to approach: District Commission: claims up to ■50 lakh. State Commission: claims ■50 lakh–■2 crore. National Commission (NCDRC): claims above ■2 crore. Step 4 — Documents needed: Invoice/bill, warranty card, photos of defective product/service, all communication with the company, delivery proof/receipt. Step 5 — You can claim: refund + replacement + compensation for mental agony + litigation costs. Tip: Many companies resolve quickly once they receive a formal legal notice — a lawyer can draft this for ■500–2000.'),
    KBEntry('co_ecommerce', 'consumer', 'E-commerce, online refund and delivery disputes', "Under Consumer Protection (E-Commerce) Rules 2020, online platforms MUST: Display seller name, address, rating, and return/refund policy clearly. Provide a grievance officer whose name and contact must be on the website. Not manipulate prices, reviews, or search results unfairly. Action for disputes: (1) Use the platform's own return/refund process first (keeps a paper trail). (2) If refused, email the company's grievance officer (check the website footer/contact page). (3) Escalate to National Consumer Helpline 1915 — they contact the company on your behalf. (4) File on e-daakhil.nic.in. Common wins: full refund for item not delivered, replacement for damaged goods, refund for counterfeit products. Time limit: act within 30 days for return/refund per most platform policies; file consumer complaint within 2 years."),
    KBEntry('co_insurance', 'consumer', 'Insurance claim denial or delay', "Insurance Regulatory and Development Authority of India (IRDAI) protects policyholders. Your rights: (1) Free look period: You can cancel a new policy within 15–30 days of receiving it for a full refund. (2) Claim settlement timelines: Life insurance — within 30 days of receiving all documents. Health insurance — cashless within 1 hour, reimbursement within 30 days. (3) Repudiation must be in writing with reasons. Action if claim is rejected or delayed: (1) Write to the insurer's Grievance Redressal Officer with all documents. (2) If unresolved in 30 days, file with the Insurance Ombudsman (online at cioins.co.in — free, no lawyer needed). (3) Alternatively file a consumer complaint at e-daakhil.nic.in. (4) Call IRDAI Bima Bharosa helpline: 155255 or 1800-4254-732."),
    KBEntry('co_misleading_ad', 'consumer', 'Misleading advertisements and false claims', "Consumer Protection Act 2019 Section 2(28): 'misleading advertisement' — falsely describes a product, gives a false guarantee, or misleads consumers about the nature, substance, quantity or quality of goods/services. CCPA (Central Consumer Protection Authority) can: issue orders to discontinue the ad, impose a fine of up to ■10 lakh on the manufacturer (■50 lakh for repeat offences), order corrective advertisements. Endorser liability: Celebrity/influencer endorsers can also be fined if they endorsed without due diligence. Complaint: File at consumerhelpline.gov.in (National Consumer Helpline 1915). ASCI (Advertising Standards Council of India): Self-regulatory body for ads — file a complaint at ascionline.in within 3 months. Online influencer ads: SEBI (for financial products), ASCI and CCPA rules require disclosure of paid partnerships — '#ad' or '#sponsored'. Food misleading claims (e.g., 'zero sugar', 'natural'): Complain to FSSAI at fssai.gov.in."),
    KBEntry('co_product_safety', 'consumer', 'Product safety and liability for defective goods', 'Consumer Protection Act 2019 introduces product liability — manufacturers, sellers and service providers are liable for harm caused by defective products. Product liability claim (Chapter VI, CPA 2019): You can claim compensation if a defective product caused personal injury, death or property damage. No need to prove negligence — strict liability applies for manufacturing defect. Action: File a complaint in the Consumer Commission against the manufacturer/seller. BIS (Bureau of Indian Standards): Certain products (helmets, electrical appliances, LPG cylinders, toys, infant milk) must carry ISI/BIS hallmark — sale without it is illegal. Complain at bisconsumer.bis.gov.in. Food adulteration: FSSAI (Food Safety and Standards Authority of India) — file a complaint at fssai.gov.in or state food safety commissioner. Drugs: Spurious or sub-standard drugs — report to the State Drug Controller or Central Drugs Standard Control Organisation (CDSCO) at cdsco.gov.in. Automobile recall: Manufacturer must replace defective parts free of cost during a recall — SIAM (siam.in) publishes recall notices.'),
    KBEntry('co_telecom_service', 'consumer', 'Telecom service complaints and TRAI regulations', "TRAI (Telecom Regulatory Authority of India) sets minimum quality of service standards for mobile and broadband providers. Your rights as a telecom consumer: 24-hour customer care mandatory. Call drop rate must be below prescribed limits. Broadband speed (minimum guaranteed speed must be disclosed; demand it in writing). Unsolicited commercial communication (spam calls/SMS): Register on DND at trai.gov.in or call 1909. If service quality is below standard: (1) First: complain to the telecom operator's customer care and get a reference number. (2) If unresolved in 30 days (for individual disputes) or as per TRAI norms: approach the Telecom Consumer Grievance Redressal Forum (CGRF) — each telecom operator has one. (3) Appeal to the Telecom Ombudsman (TDSAT) at tdsat.gov.in if CGRF decision is unsatisfactory. (4) TRAI consumer portal: trai.gov.in → Complaint Management System."),
    KBEntry('co_real_estate_rera', 'consumer', 'Real estate consumer rights — RERA', "Real Estate (Regulation and Development) Act 2016 (RERA): Protects homebuyers. Mandatory registration: All real estate projects selling more than 8 units or covering more than 500 sq m must register with the State RERA authority before launch. Buyer rights: Developer must complete project on time. Carpet area must be as promised. Delay: Builder must pay interest at SBI MCLR + 2% for every month of delay. Defect liability: 5 years from possession for structural defects — builder must rectify free of cost. False advertising: Builder cannot advertise unregistered projects. Action: File a complaint on your state's RERA portal (e.g., UP RERA, MahaRERA, RERA Kerala). Adjudicating Officer under RERA can award compensation for losses. RERA also applies to commercial units purchased for business purposes. Insolvency: If builder goes insolvent, homebuyers are classified as financial creditors under Insolvency and Bankruptcy Code — approach the NCLT."),
    KBEntry('co_food_quality', 'consumer', 'Food safety, restaurant hygiene and FSSAI rights', "Food Safety and Standards Act 2006 and FSSAI (Food Safety and Standards Authority of India) govern food quality and safety in India. Your rights: Right to safe, non-adulterated food. Right to know ingredients, allergens, and nutritional information (mandatory on packaged food labels). Restaurant hygiene: Restaurants must maintain basic hygiene standards (clean kitchen, pest-free, clean water, proper food storage). If you find a foreign object (insect, metal) in packaged/restaurant food: (1) Preserve the evidence (photo, sample in a sealed container). (2) File a complaint with the Food Safety Officer of your district. (3) File a consumer complaint at e-daakhil.nic.in. (4) File at FSSAI's consumer helpline: 1800-11-2100 (toll-free). Adulteration: Selling adulterated food can result in imprisonment of 7 years + fine under FSSAI Act. Junk food in schools: FSSAI guidelines prohibit sale of high-fat, salt, sugar (HFSS) foods within 50 metres of schools."),
    KBEntry('co_digital_service', 'consumer', 'Digital service and subscription complaints', "OTT (Netflix, Hotstar, etc.) and app subscription disputes: If charged for a subscription you did not authorise or cannot cancel: (1) Contact the platform's customer support. (2) Complain to your bank to block the recurring mandate (NACH or credit card auto-debit). (3) File a consumer complaint on e-daakhil.nic.in. Auto-renewal trap: Under IT (Intermediary) Guidelines 2021, platforms must provide clear cancellation mechanisms. Data usage charges: Telecom operators must send alerts at 80% and 100% of plan usage — failure is a TRAI violation. App store refunds: Google Play and Apple App Store have their own refund policies (typically 48 hours after purchase). Digital downloads (e-books, software) — Consumer Protection Act applies to digital goods as well (Section 2(9) definition of goods includes 'products'). National Consumer Helpline (1915): Can facilitate redressal with major OTT/telecom/e-commerce companies via their convergence mechanism."),
    KBEntry('co_vehicle_lemon', 'consumer', 'Motor vehicle defects and lemon law rights', "India does not have a specific 'lemon law' but Consumer Protection Act 2019 covers vehicle defects fully. Manufacturing defect: If a newly purchased vehicle has a recurring defect that cannot be fixed after multiple attempts, you can claim: (1) Replacement with a new vehicle of the same model. (2) Full refund of the purchase price. (3) Compensation for mental agony and associated costs. File a consumer complaint against both the dealer and the manufacturer in the District Consumer Commission. Required evidence: Purchase invoice, all service records/job cards, written communication with dealer/manufacturer. Third-party manufacturer warranty: If a car part (battery, tyre) fails under warranty, claim from the part manufacturer's warranty directly. Motor vehicle accident compensation (not consumer): handled under the Motor Vehicles Act — file a claim petition before the Motor Accidents Claims Tribunal (MACT)."),
    KBEntry('co_airline_rights', 'consumer', 'Airline passenger rights and DGCA regulations', "Directorate General of Civil Aviation (DGCA) Civil Aviation Requirement (CAR) on passenger rights: Flight delay of 2+ hours: airline must provide meals/refreshments. Delay of 24+ hours or cancellation: right to full refund or alternative flight. Denied boarding (overbooking): compensation of ■200–400 per hour of wait time, minimum ■10,000, depending on delay length and sector. Baggage lost or damaged: compensation under Montreal Convention (international) or DGCA rules (domestic) — report to the airline immediately and get a Property Irregularity Report (PIR). Action for airline consumer complaints: (1) File complaint on the airline's portal and get a reference number. (2) Use AirSewa portal (airsewa.gov.in) — DGCA's online grievance platform. (3) File a consumer complaint at e-daakhil.nic.in. Cancellation refund: Under DGCA rules, refund must be processed within 7 working days of cancellation."),
    KBEntry('co_medical_device', 'consumer', 'Medical device and pharma consumer rights', "Drugs and Cosmetics Act 1940 and CDSCO (Central Drugs Standard Control Organisation) regulate quality of medicines and medical devices. Spurious or sub-standard drug: If you suspect a medicine is fake or substandard (unusual colour/smell, packaging errors, no effect): (1) Preserve the medicine and packaging. (2) Report to your State Drug Controller or CDSCO at cdsco.gov.in (online complaint form). (3) File a consumer complaint — medicines are 'goods' under CPA 2019. Medical device failure: Pacemakers, knee implants, stents — report device malfunction or adverse event to CDSCO. Overpricing of essential medicines: NPPA (National Pharmaceutical Pricing Authority) controls prices of scheduled medicines. Report overcharging at nppaindia.nic.in. Hospital billing disputes: File a consumer complaint for overbilling or unnecessary procedures."),
    KBEntry('co_hotel_hospitality', 'consumer', 'Hotel and hospitality consumer rights', "Hotel consumer rights: Hotel cannot charge more than the declared tariff (Tourism Acts and MRP regulations). If a hotel refuses to honour a booking or charges cancellation fees beyond what was disclosed: file a consumer complaint. OYO/MakeMyTrip/Airbnb disputes: (1) Raise dispute through the platform's resolution centre. (2) If unresolved, file a consumer complaint at e-daakhil.nic.in against both the platform and the hotel. Star-rating fraud: Hotels falsely claiming 3/4/5 star ratings — report to the Tourism Department of the state and the Ministry of Tourism. Restaurant consumer rights: You are NOT obligated to pay a 'service charge' — it is illegal to make it mandatory per the CCPA 2022 guidelines. You can request the hotel to remove it from your bill. CCPA circular (July 2022) clarified: Service charge is voluntary; hotels cannot force it or deny entry if you refuse. For restaurant overcharging: file at National Consumer Helpline 1915."),
    KBEntry('co_education_service', 'consumer', 'Education services as consumer rights', "Supreme Court has held that educational institutions offering services for a fee are service providers under the Consumer Protection Act (for non-academic matters). Complaints you can file as a consumer against institutions: (1) Fee collected but course not delivered/accreditation revoked. (2) TC/certificates withheld. (3) Hostel/transportation service deficiency. (4) Fake affiliation or UGC recognition claims. Academic matters (examination results, admissions) are NOT consumer disputes — they must be challenged through the institution's own mechanism or in courts. UGC helpline: 1800-111-656. AICTE grievances: grievance.aicte-india.org. Distance learning scams: Verify UGC recognition at ugc.ac.in — unrecognised degrees are not valid for government jobs. Foreign degree validation: Association of Indian Universities (AIU) — aiu.ac.in. Coaching centre fraud: If a coaching institute promises exam success and takes a fee without delivering — file a consumer complaint."),
    KBEntry('co_water_electricity', 'consumer', 'Water and electricity consumer rights', "Electricity Act 2003: SERC (State Electricity Regulatory Commission) sets tariffs and consumer rights. Your rights as an electricity consumer: New connection within prescribed time. Meter reading at regular intervals — if estimated bills are given, demand actual reading. Excess billing: File a complaint with the distribution company; if unresolved, approach the Forum for Redressal of Consumer Grievances (FRCG) established under Electricity Act. Power outage compensation: Most state SERCs mandate compensation for extended outages. Smart meter: If a smart meter is installed, you have the right to see real-time consumption data. Water supply: Urban water supply is under Urban Local Bodies — file a complaint with your municipal corporation. Contaminated water supply: Report to the local health officer and file RTI asking for water testing results. For electricity consumer grievances: contact your state's SERC or the Electricity Ombudsman at the state level."),
    KBEntry('co_financial_services', 'consumer', 'Financial services consumer rights', "Mutual fund complaints: Complain to AMC → SEBI at scores.sebi.gov.in → SEBI's Ombudsman (SMART ODR at smartodr.in). Stock broker disputes: SEBI SCORES portal → Stock Exchange grievance mechanism → Arbitration at NSE/BSE. Demat account issues: Depository (NSDL/CDSL) has investor grievance portals — iepf.gov.in for unclaimed shares. Pension (NPS) complaints: Pension Fund Regulatory and Development Authority (PFRDA) — complaint at centralrecordkeepingagency.com or call 1800-110-708. EPF complaints: epfigms.gov.in or call 1800-118-005. Postal savings scheme disputes: Indian Post Payments Bank (IPPB) or the nearest post office. Chit fund fraud: Chit funds are regulated by states under the Chit Funds Act 1982 — report to the state Registrar of Chit Funds and file an FIR for cheating. SEBI investor education: sebi.gov.in/sebiweb/home/HomepageIndex.jsp"),
    KBEntry('co_packaged_goods', 'consumer', 'Packaged goods, MRP and weight/measure rights', "Legal Metrology Act 2009 and Legal Metrology (Packaged Commodities) Rules 2011: Every packaged commodity MUST display: MRP (Maximum Retail Price) inclusive of all taxes, net weight/quantity, manufacturer's name and address, month and year of manufacture/expiry, best before date. Selling above MRP is illegal — both the manufacturer and seller are liable. Compulsory display: Petrol pumps must display fuel price; cinema halls cannot sell packaged food above MRP. Short weight/measure: If you suspect you are getting less than the declared quantity: (1) Report to the Legal Metrology Officer of your district. (2) File a consumer complaint. Gems and jewellery: BIS hallmarking for gold is mandatory since 2021 for certain categories — gold sold without HUID (Hallmark Unique Identification) can be reported to BIS. Fuel adulteration: Report at the Petroleum Ministry's portal or to the local petroleum officer."),
    KBEntry('co_travel_tourism', 'consumer', 'Travel, tourism and tour operator complaints', "Package tour disputes: If a tour operator fails to deliver promised services (hotel quality, inclusions): (1) Document all discrepancies with photos and bills. (2) Send a written complaint to the tour operator within 30 days. (3) File at National Consumer Helpline 1915. (4) File a consumer complaint at e-daakhil.nic.in. IATA accredited travel agents: If agent disappears with your payment, report to IATA at iata.org/en/contact/customer-portal/. Visa rejection: Consulates are not consumer service providers — visa rejection cannot be contested under CPA. Seek refund under the tour operator's terms. Travel insurance claim denial: Approach the insurer's grievance officer → Insurance Ombudsman. Pilgrimage tourism disputes: IRCTC and state tourism corporations have grievance portals. Eco-tourism and adventure sports: Ensure operators have ADAI/adventure sports safety certifications — injury claims can be filed as consumer complaints."),
    KBEntry('co_lottery_prize', 'consumer', 'Lottery, prize fraud and lucky draw scams', 'Lottery regulation: In India, private lotteries are heavily regulated and online lotteries are mostly illegal. Only state government-run lotteries are legal in states that permit lotteries (Goa, Kerala, Maharashtra, West Bengal, etc.). Lottery fraud: If you receive a call/SMS/email saying you have won a lottery/prize and must pay a processing fee: This is ALWAYS a fraud. No legitimate lottery requires you to pay a fee to collect a prize. Action: (1) Do not pay any amount. (2) File a complaint at cybercrime.gov.in. (3) Report to local police. Prize bond scams: RBI stopped prize bonds in 2015 — any prize bond scheme offered now is fraudulent. Lucky draw in shopping malls: Must be registered with the state — if unregistered, report to the district administration. Online gaming prize fraud (task-based earnings): Report to cybercrime.gov.in and file an FIR. Labour & Employment 4 existing | 16 new | 20 total'),
    KBEntry('la_law', 'labour_employment', 'Wages, termination and social security laws', 'Key laws protecting workers: Code on Wages 2020: Minimum wage must be paid; wages must be paid by 7th of next month (factories) or 10th (others). Industrial Disputes Act 1947: Factories/mines/plantations with 100+ workers cannot retrench without government permission. Severance pay: 15 days salary per year of completed service. Code on Social Security 2020: Consolidates PF, ESI, gratuity, maternity benefit. Gratuity: Payable if you complete 5 years of service — 15 days salary per year. PF: Employer must deposit 12% of basic salary. You can check your PF balance at passbook.epfindia.gov.in. ESI: Applies to factories/shops with 10+ employees; covers medical, sickness, maternity benefits. Maternity leave: 26 weeks paid leave for first two children (Maternity Benefit Act 1961, amended 2017).'),
    KBEntry('la_action', 'labour_employment', 'How to file a salary, PF or termination complaint', 'For unpaid salary / wrongful termination: Step 1: Send a written demand letter to HR/employer by registered post. Step 2: File online at the Shram Suvidha Portal (shramsuvidha.gov.in) — select your state and issue type. Step 3: Walk in to the local Labour Commissioner/Inspector office with: appointment letter, salary slips, attendance records, HR messages. Step 4: If no resolution, the Labour Commissioner calls a conciliation — this is free and often resolves disputes within 3 months. For PF issues: File grievance at epfigms.gov.in or call EPF helpline 1800-118-005 (toll-free). For ESI issues: Contact the nearest ESI office or call 1800-11-3839. For gratuity: Send a written application to the employer within 30 days of leaving. If denied, file a claim before the Controlling Authority (usually Labour Commissioner).'),
    KBEntry('la_posh', 'labour_employment', 'Sexual harassment at workplace (POSH Act 2013)', 'The Prevention, Prohibition and Redressal of Sexual Harassment at Workplace Act (POSH Act) applies to ALL workplaces: offices, factories, homes (domestic workers), restaurants, hospitals, shops, schools, even digital/remote workplaces. Sexual harassment includes: unwelcome physical contact, sexually coloured remarks, showing pornography, demanding sexual favours, making threats relating to employment. Step 1: File a written complaint with the Internal Complaints Committee (ICC) of your organisation WITHIN 3 MONTHS. (If your employer has no ICC — this itself is an offence by the employer.) Step 2: The ICC must complete enquiry within 90 days. Step 3: If your organisation has fewer than 10 employees, file with the Local Complaints Committee (LCC) — at the district level under the District Officer/Collector. Step 4: If ICC/LCC is unresponsive, file a complaint with the Police (FIR under BNS) or approach the Labour Court. She-Box portal (shebox.nic.in): central government employees can file directly online. Women Helpline: 181. iCall: 9152987821.'),
    KBEntry('la_contract_gig', 'labour_employment', "Contract workers and gig workers' rights", "Contract workers (hired through a contractor): The Contract Labour (Regulation and Abolition) Act 1970 protects you: You must get the same wages as regular workers doing the same work. The principal employer is jointly liable for wages if the contractor defaults. Report violations to the Labour Commissioner. Gig/platform workers (Zomato, Ola, Uber, etc.): The Code on Social Security 2020 includes gig workers — platforms may be required to contribute to a welfare fund. Some states (Rajasthan) have enacted Gig Worker Welfare Acts. Domestic workers: There is no central domestic worker law yet, but they are covered under Minimum Wages Act in many states. For any exploitation, contact: National Domestic Workers' helpline in some states, or local NGOs (Prayas, etc.). File a complaint with the local Labour Inspector."),
    KBEntry('la_minimum_wage', 'labour_employment', 'Minimum wage rights across sectors', "Code on Wages 2020 (in force): Mandates a universal minimum wage for all employees (no exclusions). Central Advisory Board recommends national minimum wage floor — states must pay at least this. Minimum wages are notified by state governments separately for each scheduled employment (agriculture, construction, shops, etc.). Check your state's current minimum wage at the Labour Department website or shramsuvidha.gov.in. Payment of Wages Act 1936 (consolidated under Code on Wages): Wages must be paid in full — deductions only for: tax, PF, ESI, authorised advances. Wage deduction for absence: Only proportional deduction is allowed — fining workers arbitrarily is illegal. Agricultural workers: Covered by state Minimum Wages schedules — complain to the local Agricultural Labour Officer. Underpayment: File a complaint with the Deputy Labour Commissioner — they can order recovery of arrears. Penalty for non-payment: Employer can be fined up to ■50,000 + imprisonment under Code on Wages."),
    KBEntry('la_maternity', 'labour_employment', 'Maternity benefit and parental leave rights', 'Maternity Benefit Act 1961 (amended 2017): 26 weeks paid maternity leave for the first two children; 12 weeks for the third child onwards. 12 weeks for adoptive mothers and commissioning mothers (surrogacy). Applies to: All establishments with 10+ employees. Creche facility: Mandatory in establishments with 50+ employees; mother has the right to 4 visits to the creche per day. Work from home: After maternity leave, employer may allow WFH — to be agreed upon by employer and employee. Dismissal during pregnancy: Illegal — employer cannot terminate a pregnant woman or reduce her pay during maternity leave. Paternity leave: No central law; central government employees get 15 days paternity leave; many private employers have policies. Complaint mechanism: File with the Inspector of Factories/Labour Inspector for the Maternity Benefit Act violations. Penalty for employer: Fine up to ■5,000 + imprisonment up to 1 year.'),
    KBEntry('la_workmen_comp', 'labour_employment', 'Workmen compensation and workplace accident rights', "Employees' Compensation Act 1923 (renamed from Workmen's Compensation): Employers must pay compensation if an employee suffers injury, disability or death due to an accident arising out of and in the course of employment. Applicable to: all workers (except the armed forces and those under ESI). Compensation amount: For permanent total disablement: 60% of monthly wages × relevant factor (based on age). Death: 50% of monthly wages × relevant factor, minimum ■1.20 lakh. How to claim: File a claim with the Commissioner for Employees' Compensation (usually the Labour Commissioner) in the state where the accident occurred. Timeline: Employer must inform the Commissioner within 7 days of a serious accident. ESI: If worker is covered under ESI, compensation is paid through ESI — not under this Act. Contractor workers: The principal employer is also liable if the contractor fails to pay compensation. Medical expenses: Employer must pay the employee's medical expenses arising from the accident."),
    KBEntry('la_retrenchment', 'labour_employment', 'Retrenchment, closure and lay-off rights', 'Industrial Disputes Act 1947 (consolidated under Industrial Relations Code 2020 — pending state rules): Lay-off: Employer cannot lay off workers without compensation — 50% of basic wages + dearness allowance for each day of lay-off. Retrenchment (for establishments with <100 workers): 15 days notice or pay in lieu + retrenchment compensation (15 days wage per year of service). Retrenchment (for 100+ workers): Prior government permission required. LIFO principle: Last person joined must be the first retrenched (within the same category). Closure: Employer with 100+ workers must get government permission to close — 3 months notice. Wrongful termination: File complaint before Labour Court or Industrial Tribunal. Non-compete clauses post-employment: Not enforceable in India under Section 27 Contract Act (restraint of trade is void). Notice period: Typically as per employment contract — employer must pay notice pay if not served.'),
    KBEntry('la_trade_union', 'labour_employment', 'Trade union rights and collective bargaining', 'Trade Unions Act 1926 (consolidated under Industrial Relations Code 2020 — pending rules): 7 or more workers can form a trade union. Union must be registered with the Registrar of Trade Unions. Registered union: Can negotiate with management, represent workers in dispute proceedings, represent workers before Labour Courts. Unfair labour practices: Employer cannot dismiss or discriminate against workers for joining a union or participating in union activities — this is an offence. Strike rights: Workers in essential services (hospitals, water, power) cannot strike without 14-day notice. In other sectors, notice must be given. Lockout: Employer must give advance notice before a lockout — illegal lockout allows workers to claim wages. Recognition of trade unions: Collective bargaining agent is the union with more than 51% membership or as determined by secret ballot. Complaint for unfair labour practice: File with the Industrial Tribunal or Labour Court.'),
    KBEntry('la_occupational_health', 'labour_employment', 'Occupational health and workplace safety rights', "Occupational Safety, Health and Working Conditions Code 2020 (OSHWC Code): Every employer must provide a safe work environment, adequate welfare facilities, and health services. Working hours: 8 hours/day, 48 hours/week for most sectors; maximum with overtime: 9 hours/day, 60 hours/week. Overtime pay: Must be paid at DOUBLE the ordinary rate of wages. Mandatory facilities in factories: Canteen (for 250+ workers), creche (for 50+ women), first aid box, washrooms. Hazardous work: Workers in mines, construction, chemicals must receive safety training; employer must provide PPE (protective equipment) free of cost. BOCW (Building and Other Construction Workers) Act 1996: Construction workers must be registered with BOCW Welfare Board to access health insurance, accident compensation, education for children. Register at your state's BOCW portal. Complaint for unsafe conditions: File with the Factory Inspector (under Factories Act) or OSHWC Inspector."),
    KBEntry('la_pf_epf', 'labour_employment', 'PF, EPF and pension rights in detail', "Employees' Provident Fund and Miscellaneous Provisions Act 1952: Applies to establishments with 20+ employees. Both employer and employee contribute 12% of basic+DA. Employee contribution can be reduced voluntarily; employer's share is mandatory. Pension (EPS): 8.33% of employer's contribution goes to the Employees' Pension Scheme — pensionable after 10 years of service. Higher pension option: Supreme Court (November 2022) allowed employees to opt for higher pension based on actual salary — deadline passed, but claims are ongoing. PF withdrawal: Full withdrawal allowed on retirement or 2 months' unemployment. Partial withdrawal for housing, education, marriage, medical treatment — conditions apply. PF transfer: Use Form 13 or EPFO member portal (passbook.epfindia.gov.in) to transfer PF from old to new employer. Employer not depositing PF: This is a criminal offence — file complaint at epfigms.gov.in or with the Regional PF Commissioner. EPFO helpline: 1800-118-005 (toll-free)."),
    KBEntry('la_esi_health', 'labour_employment', 'ESI — medical and sickness benefits for workers', "Employees' State Insurance Act 1948 (ESI): Applies to non-seasonal factories/establishments with 10+ employees; employees earning up to ■21,000/month are covered (■25,000 for persons with disability). Benefits under ESI: (1) Medical benefit: free out-patient and in-patient treatment at ESI dispensaries and hospitals for employee AND dependants. (2) Sickness benefit: 70% of daily wages for up to 91 days of sickness per year. (3) Maternity benefit: 100% of wages for 26 weeks. (4) Disablement benefit: 90% of wages for temporary disablement; pension for permanent disablement. (5) Dependent benefit: pension to dependants in case of employee's death from employment injury. How to access: Get ESIC IP (Insured Person) number from employer; visit nearest ESIC dispensary with IP card. Employer not registering under ESI: file complaint with ESIC Regional Office. ESIC helpline: 1800-11-3839."),
    KBEntry('la_apprentice', 'labour_employment', 'Apprenticeship and intern rights', 'Apprentices Act 1961 (amended 2014): Employers with 30+ employees in certain industries must engage apprentices (2.5%–10% of total workforce depending on sector). Stipend: Minimum stipend is prescribed by the government and notified annually. Duration: 6 months to 3 years depending on the trade. National Apprenticeship Training Scheme (NATS) and NAPS portals for registration. Interns (not apprentices under the Act): No separate central law; governed by offer letter/MOU terms. Unpaid internships: Not illegal but exploitative — demand a written internship agreement. Workplace safety for interns: POSH Act applies to interns. Sexual harassment during internship: File complaint with the ICC of the organisation. Stipend disputes for statutory apprentices: file complaint with the Apprenticeship Adviser (Regional Directorate of Skill Development).'),
    KBEntry('la_remote_work', 'labour_employment', 'Work from home, remote work and digital labour rights', "There is no dedicated 'work from home' law in India; regular labour laws apply. Code on Occupational Safety, Health and Working Conditions 2020: The government has released model standing orders for gig/platform and WFH workers — states may notify specific rules. WFH employee rights: Same wages, PF, ESI as in-office employees. POSH Act applies — harassment on official digital platforms (video calls, messaging apps) is covered. Working hours: Employer cannot expect 24/7 availability; working hours must not exceed Code limits. Data privacy of employee: Employer CANNOT install tracking software without disclosure. Equipment and expenses: Best practice (and emerging consensus) is that employer must provide or reimburse equipment for WFH. Freelancers: Treated as self-employed; no labour law protections apply. Use a written contract to protect yourself. For gig workers: Code on Social Security 2020 extends some protections — check your state's gig worker welfare scheme."),
    KBEntry('la_gratuity', 'labour_employment', 'Gratuity: eligibility, calculation and claims', "Payment of Gratuity Act 1972 (consolidated under Code on Social Security 2020): Eligibility: Any employee (including contract) who has completed FIVE YEARS of continuous service. (Exception: continuous service of less than 5 years if death or disablement occurs.) Calculation: 15 days' wages × number of completed years of service ÷ 26. Maximum limit: ■20 lakh (revised; government employees have higher ceiling). Payment timeline: Within 30 days of the employee becoming eligible. How to claim: Submit Form I (application for gratuity) to the employer within 30 days of becoming eligible. Employer must send Notice of Determination within 15 days of receipt of application. If employer refuses or underpays: (1) Send a written demand by registered post. (2) File a claim before the Controlling Authority (usually Labour Commissioner) — they can order payment with interest at 10% per annum. Tax exemption: Gratuity up to ■20 lakh is tax-exempt."),
    KBEntry('la_equal_pay', 'labour_employment', 'Equal pay and gender/caste pay discrimination', "Equal Remuneration Act 1976 (consolidated under Code on Wages 2020): Employers must pay EQUAL remuneration to men and women for the same work or work of similar nature. Discrimination on grounds of sex in recruitment or working conditions is prohibited. Evidence to collect: Salary slips, appointment letters, job description comparisons. Caste-based pay discrimination: While no explicit 'equal pay for caste' law exists, SC/ST employees in government are protected by service rules; in private sector, BNS Section 153 and Constitution Article 15 (indirect application) may apply. Gender pay gap complaint: File with the Labour Inspector or Equal Remuneration Authority. Central government employees: Pay parity governed by 7th Pay Commission recommendations. Complaint: labour.gov.in or shramsuvidha.gov.in; state labour department portals. NHRC: For systemic pay discrimination affecting a protected group, file with NHRC."),
    KBEntry('la_migrant_worker', 'labour_employment', 'Inter-state migrant worker rights', "Inter-State Migrant Workmen (Regulation of Employment and Conditions of Service) Act 1979 (ISMW Act — to be replaced by OSHWC Code): Applies when workers are recruited by contractors from one state and employed in another. Rights: Must receive same wages as local workers; employer must pay displacement allowance (extra 50% of wages for first month); employer must arrange accommodation, transport, medical facilities. Registration: Contractor must register if employing 5+ inter-state migrant workers. Migrant Worker Distress Helpline: 14494 (during COVID), varies by state. e-Shram portal (eshram.gov.in): Unorganised workers including migrants can register for ■2 lakh accident insurance and access to government schemes. Minimum wage in destination state: Must be paid even if origin state's minimum wage is lower. If stranded: Contact the destination state's Labour Commissioner or the origin state government's migration cell. NDMF (National Disaster Management) and state governments have special protocols for migrant welfare in emergencies."),
    KBEntry('la_bonus', 'labour_employment', 'Bonus rights under the Payment of Bonus Act', 'Payment of Bonus Act 1965 (consolidated under Code on Wages 2020): Applies to all establishments with 20+ employees (employees earning up to ■21,000/month are eligible). Minimum bonus: 8.33% of annual salary (or ■100 whichever is higher) regardless of profit. Maximum bonus: 20% of annual salary, paid from the allocable surplus of the establishment. Eligibility: Employee must have worked for at least 30 working days in the accounting year. Payment deadline: Bonus must be paid within 8 months of the close of the accounting year. New establishments: Exempt from bonus for the first 5 years. If employer does not pay bonus: (1) Send a written reminder. (2) File a complaint with the Labour Inspector/Commissioner. Bonus dispute: Referred to Labour Court. Deduction of bonus: Employer cannot deduct bonus for misconduct unless formal disciplinary proceedings have been concluded.'),
    KBEntry('la_leave_entitlement', 'labour_employment', 'Leave entitlements: EL, CL, SL and national holidays', 'Leave entitlements in India vary by state and industry, but broadly under the Factories Act / Shops & Establishments Acts: Earned Leave (EL): 1 day for every 20 days worked (factories); minimum 15 days per year. Can be accumulated (usually up to 45 days). Casual Leave (CL): Usually 10–12 days per year for unexpected/personal reasons. Generally cannot be carried forward. Sick Leave (SL): 6–12 days per year; medical certificate required for extended sick leave. National and Festival holidays: All employers must give paid leave on the 3 national holidays (Republic Day, Independence Day, Gandhi Jayanti) + state-declared festival holidays. Maternity leave: 26 weeks (see la_maternity entry). Compensatory off: For working on holidays — employer must grant comp-off or pay double wages. If leave is wrongfully denied or encashment withheld: File a complaint with the Labour Commissioner. Leave encashment: Accumulated EL can be encashed at retirement — up to 300 days is tax-exempt for central government employees.'),
    KBEntry('la_sexual_harassment_men', 'labour_employment', 'POSH Act — coverage for all genders and informal workplaces', "POSH Act 2013: While the Act mentions 'women' as the complainant, the Supreme Court and several High Courts have interpreted it broadly to protect all genders in the spirit of dignity at work. Formal workplaces (10+ employees): Must have an Internal Complaints Committee (ICC) with: minimum 4 members, presided by a senior woman employee, at least one external member from an NGO. Informal workplaces (<10 employees), domestic workers, agriculture workers: File with the Local Complaints Committee (LCC) at the district level under the District Officer. Process: Written complaint within 3 months of the incident. ICC must complete inquiry within 90 days and submit report within 10 more days. Interim relief during inquiry: ICC can recommend transfer of the respondent/complainant, or grant leave. If the employer does not have an ICC: This is itself an offence (fine up to ■50,000 for the employer). False complaint: If found malicious, complainant can be penalised — but mere inability to prove a complaint is NOT malicious. She-Box: For central government employees at shebox.nic.in. Civil, Property & Contract Law 3 existing | 17 new | 20 total"),
    KBEntry('pr_law', 'civil_property', 'Property, rent and tenancy law', "Key laws: Transfer of Property Act 1882: governs buying/selling/leasing property. Any transfer of immovable property above ■100 must be in writing and registered. Registration Act 1908: sale deeds and long-term leases (over 11 months) must be registered at Sub-Registrar's office. Stamp Duty: Each state sets stamp duty rates on property transactions. Unregistered documents may not be admissible in court. State Rent Control Acts: protect tenants from arbitrary eviction. Common protections: Landlord cannot evict without a court order. Eviction only on specific grounds: non-payment of rent, subletting without permission, personal need, structural damage. Model Tenancy Act 2021 (adopted by some states): limits security deposit to 2 months' rent (residential). Specific Relief Act 1963: a party can get a court order to enforce a specific contract (e.g., force seller to complete a sale)."),
    KBEntry('pr_deposit_eviction', 'civil_property', 'Security deposit not returned and eviction disputes', "Security deposit rights: The Model Tenancy Act 2021 caps security deposit at 2 months' rent (residential) and 6 months' rent (commercial). Landlord MUST return deposit within 1 month of vacating (after deducting legitimate damages with receipts). What to do if deposit not returned: Step 1: Send a written demand notice by registered post with acknowledgement due (RPAD) — give 15 days. Step 2: File a complaint with the Rent Authority/Rent Tribunal (established under Model Tenancy Act in states that adopted it). Step 3: Alternatively, file a consumer complaint (if it was a service-provider relationship) or a civil suit for recovery. For illegal eviction: Landlord CANNOT lock you out, cut electricity/water, or remove belongings without a court order. This is illegal — file an FIR for criminal trespass (BNS Section 329). Get an injunction from the Civil Court to restore possession. Evidence to keep: signed agreement, rent receipts, move-in/move-out photos, all communication."),
    KBEntry('pr_property_fraud', 'civil_property', 'Property fraud and land grabbing', "Common property frauds: forged sale deeds, impersonation of owner, fraudulent power of attorney, double sale (selling to two buyers), encroachment, benami transactions. Prevention: (1) Always verify ownership at the Sub-Registrar's office before buying/renting. (2) Verify Encumbrance Certificate (EC) — shows all registered transactions on a property. (3) Check property records on your state's land record portal (e.g., Bhulekh UP, Bhoomi Karnataka, Dharani Telangana). If you suspect fraud: (1) File an FIR at the police station (BNS cheating provisions). (2) File a complaint with the Registrar's office where the forged document was registered. (3) Approach the civil court for cancellation of fraudulent documents. (4) Benami properties (held in another's name to hide black money) can be reported to the Benami Prohibition Unit of Income Tax."),
    KBEntry('pr_will_succession', 'civil_property', 'Wills, succession and inheritance rights', 'Indian Succession Act 1925: Governs succession for Christians, Parsis, and for civil marriages. Hindu Succession Act 1956 (amended 2005): Governs Hindus, Sikhs, Jains and Buddhists. Daughters now have equal coparcenary rights in ancestral property — right of birth. Muslim succession: Governed by Muslim Personal Law (Shariat). Application Act 1937. Will: A person of sound mind, age 18+, can make a will. No stamp duty or registration required (recommended but not mandatory). Probate: In some states, courts require a probate certificate (granted by High Court) to give effect to a will. Intestate succession: If a person dies without a will — property devolves as per the relevant personal law. If a will is disputed: File a probate case or a civil suit in the District Court. Nominee vs legal heir: Bank account nominee has the right to receive money, but is a trustee — must ultimately pass on assets to legal heirs. Succession certificate: Required for movable property (bank accounts, shares) — apply in civil court.'),
    KBEntry('pr_ancestral_property', 'civil_property', 'Ancestral property and coparcenary rights', "Hindu Undivided Family (HUF) property: Property inherited through four generations without partition is 'ancestral property'. Coparcenary: All direct descendants in the male line (and daughters since 2005 amendment) have an equal right by birth. Daughter's rights (2005 amendment, confirmed by Supreme Court 2020): Daughters are coparceners from birth — equal rights even if father died before 2005. Partition of HUF property: Any coparcener can demand partition. File a partition suit in the Civil Court. Self-acquired property of father: Not ancestral — father can will it to anyone; children have no right by birth. Benami property held in a child's name by parent: Parent's own funds = not the child's property; child's own funds = child's property. Nominee in HUF: HUF property does not pass by nomination — it passes by law. Revenue records: Ensure your name is mutated in revenue records after inheriting property — approach the local tehsildar/revenue office with death certificate and legal heir certificate."),
    KBEntry('pr_mutation', 'civil_property', 'Property mutation and revenue record updates', "Mutation (dakhil-kharij): The process of updating the ownership name in the land revenue records (khatauni/khata) after a sale, inheritance, gift or court order. Why it matters: Mutation does not create title but is needed for paying property tax and is evidence of possession. Process varies by state — check your state's revenue department portal (e.g., Bhulekh UP, Bhoomi Karnataka, Dharani Telangana). Documents typically required: Sale deed / Will + Probate / Court decree + death certificate, identity proof, original title documents. Time limit: File mutation application within the time prescribed by your state revenue law (typically within 3–6 months of transaction). If mutation is delayed: File RTI asking for status; complain to the Sub-Divisional Magistrate (SDM). Encumbrance Certificate (EC): A certificate from the Sub-Registrar's office showing all registered transactions on a property — essential for verifying title before purchase. Patta (in South India): Document issued by revenue authority confirming land possession. If wrong mutation done by fraud: File a revision petition before the Revenue Commissioner or civil court."),
    KBEntry('pr_commercial_lease', 'civil_property', 'Commercial property and business lease rights', "Commercial lease: Governed by Transfer of Property Act 1882 + the specific lease deed + state Rent Control Acts (for older commercial tenancies). Registration: Leases exceeding 11 months must be registered at the Sub-Registrar's office — unregistered commercial leases are not valid for more than 11 months. Lock-in period: If the lease has a lock-in clause, neither party can terminate early without penalty — courts enforce this. Rent increase: As per lease deed — typically with a 5–15% annual escalation clause. Maintenance and repairs: Unless agreed otherwise, major structural repairs are the landlord's responsibility. Security deposit (commercial): No statutory cap — governed by the lease deed. Eviction of commercial tenant: Must follow legal process — cannot lock out without court order. Eviction for non-payment of rent: File an eviction suit in the appropriate civil court. Stamp duty on commercial lease: Applicable in most states — check your state's stamp duty schedule. RERA: Does not apply to commercial leases (only to sale of commercial units in real estate projects)."),
    KBEntry('pr_builder_dispute', 'civil_property', 'Builder-buyer disputes and RERA remedies', 'RERA (Real Estate Regulation and Development Act 2016): Builder must register all projects with the State RERA authority before marketing. Buyer rights: (1) Developer must complete project as per the registered plan — no changes without buyer consent. (2) Delay: buyer can withdraw and get full refund with interest, OR continue and receive delay compensation. (3) Structural defect liability: 5 years from possession. (4) False advertising: Developer cannot advertise unregistered projects. HOW TO FILE: Go to your State RERA portal (e.g., maharerait.mahaonline.gov.in, rera.up.nic.in). Under RERA, adjudicating officer can award compensation; Appellate Tribunal can be approached if unsatisfied. NCLT: If the builder has gone under insolvency, homebuyers (treated as financial creditors under IBC) file claims before the NCLT. Consumer Commission: Homebuyers can file at Consumer Commission as an alternative to RERA — courts have held both remedies available. Class action: RERA allows multiple complainants to file together against the same project.'),
    KBEntry('pr_joint_ownership', 'civil_property', 'Joint property ownership and co-owner disputes', "Types of joint ownership in India: Joint tenancy: Co-owners have equal undivided shares with right of survivorship (common in HUF property). Tenancy in common: Co-owners can have unequal shares; no right of survivorship — each owner's share passes to their heirs. Dispute between co-owners: Any co-owner can file a suit for partition in the Civil Court. Sale by one co-owner without others' consent: The purchaser gets only the selling co-owner's share — they become a co-owner, not full owner. If a co-owner denies you access to the jointly owned property: File a civil suit for possession/injunction. Income from joint property: All co-owners are entitled to their proportionate share of rental income. Tax: Rental income is taxable in proportion to each owner's share. Property inherited by multiple siblings: Each sibling has rights — partition by agreement (registered partition deed) is the cleanest resolution; if disagreement, file partition suit."),
    KBEntry('pr_encroachment', 'civil_property', 'Land encroachment and illegal possession', "Encroachment: Unauthorised occupation of another person's land or construction on boundary beyond one's land. Steps to address encroachment: (1) Identify the exact boundary using revenue survey records (Khasra/Patta/Khatauni), FMB (Field Measurement Book) or the services of a licensed surveyor. (2) Send a legal notice to the encroacher by registered post. (3) File a civil suit for mandatory injunction and possession in the District Court. If the encroachment is on government land: Report to the local tehsildar or district collector — they can initiate eviction proceedings. BNS Section 329 (Criminal trespass): If someone is forcibly occupying your property, file an FIR. Limitation period: A civil suit for recovery of possession of immovable property must be filed within 12 years of the encroachment — after that, adverse possession may apply. Adverse possession: Continuous, peaceful, open possession for 12 years without the owner's permission gives the possessor a right to claim title — defend against this by filing suit early. Survey: Approach the district Survey Office for official demarcation."),
    KBEntry('pr_gifting_property', 'civil_property', 'Gift deed and property transfer by gift', "Transfer of Property Act 1882, Section 122: A 'gift' is a voluntary transfer of movable or immovable property without consideration. Gift deed for immovable property: Must be in writing, signed and attested by two witnesses, and REGISTERED at the Sub-Registrar's office — otherwise void. Stamp duty: Applicable on gift deeds (state-specific rates; often lower than sale deed if between blood relatives). Gift to minor: Accepted on the minor's behalf by a guardian. Revocation of gift: A gift once made cannot be revoked at will unless: the gift deed specifies conditions under which it can be revoked, or the donee refuses to comply with a condition (onerous gift). Senior citizen / Parent protection: Maintenance and Welfare of Parents and Senior Citizens Act 2007 — if a parent gifts property to children who then neglect them, the gift can be cancelled by the Maintenance Tribunal. Will vs Gift: Gift takes effect immediately; Will takes effect only after the death of the testator. Gift tax: No gift tax in India since 1998; however, gifts received (above ■50,000 from non-relatives) are taxable as income."),
    KBEntry('pr_tenancy_agreement', 'civil_property', 'Rental agreement, rights and dispute resolution', 'Rent/Lease Agreement: Agreements for 11 months or less need not be registered (but should be notarised). Agreements beyond 11 months must be registered — stamp duty applies. Landlord obligations: Provide a habitable property; maintain structural safety; not interfere with quiet enjoyment. Tenant obligations: Pay rent on time; not sublet without permission; not damage the property. Security deposit: Model Tenancy Act 2021 caps it at 2 months (residential) / 6 months (commercial). Rent receipt: Tenant has the right to demand a rent receipt for every payment. Notices before eviction: Minimum notice period is usually as per the State Rent Control Act (30–90 days depending on the state and type of tenancy). Illegal eviction (lockout, cutting utilities): Criminal trespass (BNS Section 329) — file FIR and seek civil court injunction. Online dispute resolution: Some state RERA authorities and rent tribunals accept online filings. Mediation: Many landlord-tenant disputes can be resolved through Lok Adalat — approach DLSA.'),
    KBEntry('pr_purchase_verification', 'civil_property', 'Property due diligence before purchase', "Essential title verification steps: (1) Check the Encumbrance Certificate (EC) for the last 30 years — obtainable from Sub-Registrar's office or state portal. EC shows all mortgages, leases, sale deeds. (2) Verify ownership by examining the original title documents (sale deed chain going back 30 years). (3) Check the land use (agriculture/residential/commercial) on the master plan of the local authority (municipality/development authority). (4) Verify that all property taxes are paid — obtain a no-dues certificate from the local body. (5) Check for any pending litigation — search at the local civil court. (6) Ensure the seller is the ONLY owner — check for other co-owners, HUF claims, court orders. (7) Search for benami transactions — income tax department's Benami Prohibition Unit. For apartments/flats: Ensure building plan is sanctioned by the local authority. Occupation Certificate (OC) and Completion Certificate (CC) must be obtained by builder. RERA registration: Verify the project is RERA registered at your state's RERA portal."),
    KBEntry('pr_nri_property', 'civil_property', 'NRI property rights and inheritance in India', "NRIs (Non-Resident Indians) can: Buy residential and commercial property in India (no RBI permission needed). Cannot buy agricultural land, plantation property or farmhouse (except by inheritance or gift from a resident Indian). FEMA (Foreign Exchange Management Act 1999): All NRI property transactions must comply with FEMA. Rental income from Indian property: Taxable in India (20% TDS to be deducted by tenant). Capital gains on sale: Taxable in India; DTAA (Double Taxation Avoidance Agreement) may reduce tax. Power of Attorney: NRI can execute a PoA in favour of a resident to manage/sell property — must be notarised and apostilled in the country of residence, then registered in India. Property disputes involving NRI: Indian courts have jurisdiction if the property is in India. Inheritance: NRI can inherit any type of property in India. Repatriation: Up to USD 1 million per year can be repatriated from sale of inherited property — subject to tax compliance. NRI helpdesk: pravaseebharathiya.gov.in and MEA's eMigrate portal."),
    KBEntry('pr_mortgage_loan', 'civil_property', 'Property mortgage, home loan rights and SARFAESI', "Mortgage: Pledging property as security for a loan. Must be registered if it involves transfer of title (equitable mortgage is created by deposit of title deeds). Home loan rights: You have the right to a complete copy of all loan documents. Pre-payment is allowed (no pre-payment penalty on floating-rate home loans — RBI rule). SARFAESI Act 2002: Allows banks/NBFCs to seize and sell mortgaged property without court order for NPA (non-performing) accounts. SARFAESI safeguards: Bank must give 60 days' notice before taking possession. Borrower can file an objection within 45 days. Debt Recovery Tribunal (DRT): Borrower can approach the DRT to contest SARFAESI action. Personal guarantee: If you are a personal guarantor for another's loan, you are equally liable — the bank can invoke your personal assets. If bank charges wrong prepayment penalty: Complain to RBI Ombudsman at cms.rbi.org.in. Interest rate switchover: You have the right to switch from floating to fixed rate (bank may charge a small fee)."),
    KBEntry('pr_partition_suit', 'civil_property', 'Partition suit and division of joint property', 'When co-owners cannot agree on division of property, any co-owner can file a partition suit in the Civil Court of the district where the property is located. Process: (1) File the suit with list of co-owners and shares. (2) Court issues notice to all co-owners. (3) Court may appoint a Commissioner to prepare a scheme of partition. (4) Final decree of partition. Preliminary decree: States entitlement of each party. Final decree: Specifies the exact portion allotted. Partition by consent: If all agree, they can execute a registered Partition Deed — cheaper and faster than a suit. Limitation: No limitation period for partition of joint property (as long as the joint possession continues). Expenses: Court fees on the value of the property; but generally lower than a recovery suit. Ancestral vs self-acquired: Court will first determine whether the property is ancestral (all coparceners share) or self-acquired (only the owner decides). Female coparcener: Can file for partition and demand physical division of the ancestral property (confirmed by Supreme Court 2020).'),
    KBEntry('pr_adverse_possession', 'civil_property', 'Adverse possession and limitation for property suits', "Adverse possession: A person who has been in continuous, open, hostile, exclusive and peaceful possession of someone else's land for 12 years (30 years for government land) can claim ownership by adverse possession. Requirements: Possession must be actual, visible, continuous for the limitation period, and WITHOUT the owner's permission. If you are the original owner: File a civil suit for recovery of possession BEFORE 12 years expires. If you are the possessor (claiming adverse possession): (1) Consult a lawyer — this is a complex area with recent Supreme Court judgments tightening the test. (2) Apply for mutation in revenue records to establish evidence of possession. Government property: Adverse possession cannot be claimed against government land (30-year period in any case). The Limitation Act 1963 governs limitation periods for property suits. Key rule: Not filing suit within the limitation period extinguishes the right — always act promptly when you discover encroachment."),
    KBEntry('pr_housing_society', 'civil_property', 'Housing society rights and cooperative disputes', "Cooperative Housing Societies are registered under state Cooperative Societies Acts. Member rights: Right to receive a copy of the society's bye-laws, audited accounts, and minutes of general meetings. Right to vote in elections and stand for the managing committee. Maintenance charges: Society can only levy charges as per registered bye-laws. Arbitrary charges can be challenged. Parking: Society cannot allocate/sell parking spots in violation of the approved building plan. NOC for transfer: Society must give NOC for flat transfer within prescribed time (30–60 days in most states). Discrimination: Society cannot deny membership or amenities based on religion, caste, marital status (some landmark cases under Maharashtra cooperative law). Complaints: File with the Registrar of Cooperative Societies of your state. Disputes: Cooperative dispute (between member and society) goes to the Cooperative Court (not regular civil court). Major repairs and redevelopment: Members can approach the state cooperative authorities if the society refuses necessary repairs or redevelopment."),
    KBEntry('pr_land_acquisition', 'civil_property', 'Land acquisition and compensation rights', 'Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act 2013 (LARR Act): For government acquisition: Mandatory social impact assessment, prior consent of 70% (private project) or 80% (PPP project) of affected families. Compensation: Market value × 1 (urban) or × 2 (rural); plus solatium (100% of market value); plus multiplier — total can be 2–4× market value. Rehabilitation and Resettlement (R&R;): Entitled to alternative land, employment, housing, and other benefits. Challenge: If compensation is inadequate, appeal to the Land Acquisition Collector → Reference to Civil Court under Section 64 → High Court. Emergency acquisition: Urgency clause (Section 40) must be used sparingly — challenge if misused. Acquired land not used for stated purpose: Owner can reclaim the land (by Supreme Court ruling). Tribal and forest land: Additional protections under Forest Rights Act and PESA — prior consent of Gram Sabha required. NALSA legal aid: Free legal representation for persons whose land is being acquired — call 15100.'),
    KBEntry('pr_stamp_duty_reg', 'civil_property', 'Stamp duty, registration and transaction costs', 'Stamp Act 1899 (central) + State Stamp Acts: Stamp duty is payable on legal documents at the time of execution. Registration Act 1908: Compulsory registration for sale deeds, mortgages, gifts, partition deeds of immovable property. Stamp duty rates vary by state and type of document (typically 3–7% for residential property transactions). Registration fee: Usually 1% of property value, subject to maximum (varies by state). Important: Circle rate (government guideline value) — stamp duty is calculated on the higher of circle rate or actual transaction value. Online payment: Most states now allow online stamp duty payment (e.g., GRAS in Maharashtra, SHCIL in many states). Unstamped/under-stamped documents: Inadmissible as evidence in court until proper stamp duty (with penalty) is paid. Rectification deed: If an error is made in a registered document, a registered rectification deed can correct it. Registration must be done within 4 months of the date of the document. Banking & Financial Services 3 existing | 16 new | 19 total'),
    KBEntry('ba_law', 'banking_finance', 'RBI rules on unauthorised transactions', "RBI's Customer Liability Framework (Circular DBR.No.Leg.BC.78/09.07.005/2017-18): ZERO liability if: Bank negligence (no fault of customer) — regardless of when reported. ZERO liability if: Third-party breach (not customer's fault) and reported within 3 working days of bank's SMS alert. Limited liability if: Reported between 4–7 working days — capped at ■5,000–■25,000 depending on account type. Unlimited liability if: Customer's own negligence (shared OTP, PIN, credentials). Bank must credit the disputed amount to your account within 10 working days of reporting. Resolution timeline: Bank must resolve within 90 days. If your bank does not follow RBI rules — complain to the Banking Ombudsman (free, quick)."),
    KBEntry('ba_action', 'banking_finance', 'Banking fraud: immediate action steps', "Step 1 — WITHIN MINUTES: Call the bank's 24x7 helpline. Block your card/account. Note the complaint reference number. Every bank has a helpline: SBI 1800-425-3800, HDFC 1800-202-6161, ICICI 1800-1080, Axis 1800-419-5959. Step 2 — SAME DAY: Send a written email to the bank's official email with: transaction details, UTR number, date/time. Step 3 — Within 3 working days: File a formal written complaint with the bank branch. Get acknowledgement. Step 4 — For UPI fraud: ALSO report at cybercrime.gov.in and call 1930. Step 5 — If bank does not resolve within 30 days or gives unsatisfactory response: File with Banking Ombudsman via RBI's CMS portal: cms.rbi.org.in (free, no lawyer needed). Step 6 — If Ombudsman doesn't help: Approach the Consumer Court or RBI's Appellate Authority. Evidence to keep: Bank statements, SMS/email alerts, transaction receipts, call recordings."),
    KBEntry('ba_loan_recovery', 'banking_finance', 'Loan recovery harassment and your rights', "RBI Fair Practices Code for loan recovery agents: Recovery agents CANNOT: call before 8 AM or after 7 PM. Use abusive/threatening language. Harass family members who are not guarantors. Visit your workplace without your consent. Seize assets without court order (for most unsecured loans). Your rights when under loan stress: (1) You can request a loan restructuring or moratorium — banks must consider this per RBI guidelines. (2) Contact the bank's grievance redressal officer first. (3) If harassment continues, file a police complaint for criminal intimidation (BNS Section 296). (4) File with the Banking Ombudsman (cms.rbi.org.in) citing RBI Fair Practices Code violation. For NBFC (non-bank finance company) harassment: Complain to RBI at sachet.rbi.org.in. IMPORTANT: Even in genuine default, a lender cannot seize movable property without a court order (for unsecured loans)."),
    KBEntry('ba_credit_score', 'banking_finance', 'Credit score, CIBIL and credit report rights', "India has four credit bureaus: CIBIL (TransUnion), Experian, Equifax, CRIF High Mark. CIBIL score range: 300–900; score of 750+ is generally considered good for loan eligibility. Free credit report: Every person is entitled to ONE free credit report per year from each bureau — request at cibil.com, experian.in, equifax.co.in, crifhighmark.com. Checking your own score does NOT lower it (this is a 'soft enquiry'). Error in credit report: (1) File a dispute on the bureau's website — they must respond within 30 days. (2) If unresolved, file a complaint with RBI at cms.rbi.org.in. Common errors: wrong late payment marks, accounts you never opened (could indicate identity theft), duplicate entries. Improving CIBIL score: Pay all EMIs on time; reduce credit card utilisation below 30%; do not apply for multiple loans simultaneously. Wilful defaulter: Banks report habitual defaulters to credit bureaus — removal from the list requires settlement of dues and court order in some cases."),
    KBEntry('ba_digital_payment_safety', 'banking_finance', 'Safe digital payment practices and dispute resolution', "UPI, NEFT, RTGS, IMPS, card payments — governed by NPCI and RBI. UPI safety rules: Never share your UPI PIN with anyone. Collecting money NEVER requires entering your PIN. Charge-back rights: For fraudulent card transactions, you can file a 'dispute' or 'chargeback' with your bank within 30 days (international) or as per bank policy (domestic). Zero liability: Under RBI guidelines, if you report a fraudulent card transaction within 3 working days and it was not your negligence, liability is zero. Transaction failure: If money is debited but not credited: (1) Wait 24 hours — most auto-reverse. (2) If not resolved, file a complaint with your bank. Bank must resolve within 7 working days and compensate ■100/day for delay. (3) Escalate to RBI Ombudsman at cms.rbi.org.in. BBPS (Bharat Bill Payment System): For utility bill payment disputes — approach NPCI at npci.org.in. Safe practice: Never make payments on public Wi-Fi; always verify payee details before transfer."),
    KBEntry('ba_insurance_life', 'banking_finance', 'Life insurance rights and claim settlement', "Life insurance is regulated by IRDAI. Key rights of policyholders: (1) Free-look period: 15–30 days to cancel a new policy for a full refund. (2) Grace period: 15–30 days after premium due date before policy lapses — death benefit still payable during grace period. (3) Revival: Lapsed policy can be revived within 2–5 years by paying arrears + interest. (4) Claim settlement: Must be settled within 30 days (simple) or 90 days (investigation needed) from receiving all documents. If claim is rejected: (1) File a complaint with the insurer's Grievance Officer. (2) If unresolved in 30 days: Insurance Ombudsman at cioins.co.in (free, covers claims up to ■50 lakh). (3) Consumer Commission: If Ombudsman not satisfactory. Nominee rights: Nominee has the right to receive the claim amount on the insured's death. If no nomination, legal heirs can claim with succession certificate. Mis-selling complaint: File with IRDAI at igms.irda.gov.in or call Bima Bharosa 155255."),
    KBEntry('ba_nbfc_regulation', 'banking_finance', 'NBFC rights and shadow banking protections', 'NBFCs (Non-Banking Financial Companies): Registered with RBI under RBI Act 1934. They can lend but cannot accept demand deposits like banks. Types: NBFC-MFI (microfinance), NBFC-P2P (peer-to-peer lending), Housing Finance Companies, Gold Loan NBFCs, etc. Borrower rights from NBFCs: (1) Must receive a copy of the loan agreement, repayment schedule, and Key Facts Statement (KFS). (2) Interest rates must be transparent — no hidden charges. (3) Recovery agents must follow the same code of conduct as bank agents. Illegal NBFC: Check if the NBFC is RBI-registered at rbi.org.in/Scripts/BS_NBFCList.aspx before borrowing. Sachet portal: sachet.rbi.org.in — for complaints against NBFCs and unlicensed lenders. P2P lending: RBI-registered P2P platforms limit total lending and borrowing — maximum ■50 lakh per person. Housing Finance Companies (HFCs): Regulated by National Housing Bank — complaints at nhb.org.in.'),
    KBEntry('ba_income_tax', 'banking_finance', 'Income tax rights and dispute resolution', 'Income Tax Act 1961: Every person (resident) earning above the basic exemption limit must file an ITR. Basic exemption: ■2.5 lakh (old regime), ■3 lakh (new regime — default from FY 2024-25). Tax refund: If TDS exceeds your tax liability — file ITR to claim refund. Refunds are credited to the linked bank account. If refund is delayed: Check status at incometaxindiaefiling.gov.in; call Aaykar Sampark Kendra 1800-103-0025. Income tax notice: DO NOT IGNORE — respond within the deadline. Types: Section 139(9) (defective return), 143(1) (intimation), 143(2) (scrutiny), 148 (reassessment). Scrutiny: Cooperate and submit requested documents. If you disagree with the assessment, file an appeal before the CIT(A) within 30 days. Second appeal: Income Tax Appellate Tribunal (ITAT). PAN-Aadhaar linking: Mandatory — penalty ■1,000 if not done; income tax consequences for non-PAN transactions.'),
    KBEntry('ba_gst_consumer', 'banking_finance', 'GST rights for consumers and small businesses', 'Goods and Services Tax (GST) — Consumer rights: Every service provider / retailer must give you a tax invoice (for supplies above ■200 or on request). Anti-profiteering: Businesses must pass on GST reduction benefits to consumers — NAA (National Anti-Profiteering Authority) enforces this. Complaint for not passing on GST reduction: File at antiprofiteering.cbic.gov.in. GST refund for exported goods/services: Exporters can claim refund of GST — file on GST portal (gst.gov.in). Small business rights: If your annual turnover is below ■20 lakh (■10 lakh for special category states), GST registration is not mandatory. GST notice: Do not ignore — respond within 30 days. Disputes: GST Appellate Authority → GST Appellate Tribunal (being set up) → High Court. GST helpdesk: 1800-103-4786. Input Tax Credit (ITC) denial: Challenge at the GST Commissioner level — ensure your supplier has filed GSTR-1 correctly.'),
    KBEntry('ba_senior_banking', 'banking_finance', 'Banking rights for senior citizens', "RBI guidelines for senior citizen banking: Every bank must have a dedicated counter or priority service for senior citizens (75+). Senior citizens (60+) are entitled to slightly higher interest rates on FDs (0.25–0.50% higher than regular rates). Doorstep banking: Banks must provide basic banking services (cash delivery, cheque pick-up, account statements) at home for senior citizens. If denied doorstep banking: Complain to the bank's grievance officer, then RBI Ombudsman. Senior Citizen Savings Scheme (SCSS): Government scheme offering higher interest; eligible for 60+ (or 55+ on retirement). Power of Attorney (PoA) abuse: If a family member or caregiver is misusing your PoA to withdraw money: (1) Contact your bank branch immediately to revoke the PoA. (2) File a police complaint for criminal breach of trust. Elderline helpline: 14567 (24x7). Senior Citizens Financial Fraud: also report to cybercrime.gov.in."),
    KBEntry('ba_locker_facility', 'banking_finance', 'Bank locker rights and liability', "RBI circular (August 2021, revised): Banks must enter into a locker agreement with customers and cannot deny lockers unfairly. Bank liability: (1) If the bank's negligence (e.g., flood, fire, theft) leads to loss of locker contents: bank is liable up to 100× annual locker rent. (2) If due to an act of God (earthquake, lightning): bank is not liable. (3) Customers are advised NOT to keep highly valuable items or cash in lockers — banks' liability does not extend to currency. Nomination: Register a nominee for your bank locker — nominee can access the locker after your death for listing and removing contents. Unclaimed lockers: Bank can break open and inventory an unclaimed locker after 3 years — with proper notice and procedures. Right to know locker status: You can ask the bank for your locker statement/access log. Forced surrender of locker: Bank must give 3 months' notice and must arrange alternative or return all contents. Complaint for locker breach: Approach bank → RBI Banking Ombudsman."),
    KBEntry('ba_pm_schemes', 'banking_finance', 'Government financial schemes and entitlements', 'PMJDY (Pradhan Mantri Jan Dhan Yojana): Zero-balance savings account with ■1 lakh accident insurance (upgraded to ■2 lakh for Rupay card holders). Open at any bank or post office. PMSBY (Suraksha Bima Yojana): Annual accidental death/disability insurance of ■2 lakh at ■20/year premium — opt in via your bank. PMJJBY (Jeevan Jyoti Bima Yojana): ■2 lakh life insurance at ■436/year — opt in via your bank. APY (Atal Pension Yojana): Guaranteed pension of ■1,000–5,000/month at age 60 — enrol via bank. PM Mudra Yojana: Collateral-free loans for small businesses — Shishu (up to ■50,000), Kishore (■50,000–5 lakh), Tarun (■5–10 lakh). Apply at any bank. Stand Up India: ■10 lakh–1 crore loans for SC/ST and women entrepreneurs. Apply at nationalfinancialswitch.org. CGTMSE: Credit guarantee for MSMEs — allows banks to give loans without collateral. Pradhan Mantri Kisan Samman Nidhi (PM-KISAN): ■6,000/year to eligible farmers in three instalments — register at pmkisan.gov.in.'),
    KBEntry('ba_chit_fund', 'banking_finance', 'Chit funds — rights and fraud protection', 'Chit Funds Act 1982: Regulates chit fund companies — they must be registered with the State Registrar. Legal chit funds: Must be registered, have a foreman, hold auctions as per rules, maintain proper accounts. Illegal chit fund schemes (ponzi disguised as chit): Promise guaranteed high returns without auctions, no registration. Warning signs: Guaranteed above-market returns; no proper documentation; pressure to invest more or recruit others. If you suspect a fraudulent chit fund: (1) File a complaint with the State Registrar of Chit Funds. (2) File an FIR — offence under Prize Chits and Money Circulation Schemes (Banning) Act 1978 and BNS Section 316 (cheating). (3) Report to the state Economic Offences Wing (EOW). If a registered chit fund defaults: File a complaint with the Registrar; approach a civil court for recovery. Legitimate chit funds: Shriram Chits, Mysore Sales International (MSIL) are examples of large registered chit companies — look for registration certificates before joining.'),
    KBEntry('ba_estate_planning', 'banking_finance', 'Estate planning, nomination and digital assets', "Nomination in financial accounts: Bank accounts, FDs, mutual funds, demat accounts, PPF — register a nominee to ensure smooth transfer of assets on death. Nominee can claim assets without probate or succession certificate for most financial instruments. However: Nominee is a TRUSTEE — must distribute the asset to legal heirs as per inheritance law. Will vs Nomination: A Will can override the legal heir entitlement of a nominee (for movable property, nominee's right is paramount in practice). Digital assets: Passwords, crypto, online accounts — document them securely and inform your nominee/executor. India has no specific digital asset succession law yet. Unclaimed deposits: After 10 years of inactivity, bank deposits are transferred to DEAF (Depositor Education and Awareness Fund) — claim at iepf.gov.in or your bank. EPF nomination: Update nominee on UAN portal — very important, as EPF claim without nominee can take years. PPF nomination: Register nominee at the post office or bank branch where the account is held."),
    KBEntry('ba_insurance_motor', 'banking_finance', 'Motor insurance rights and accident claims', 'Motor Vehicles Act 1988: Third-party insurance is MANDATORY for all vehicles. Driving without valid insurance attracts fine and imprisonment. Third-party insurance: Covers injury/death to others and damage to their property — claim paid from the insurer directly; you as the insured are not personally liable to the injured party up to the insured sum. Own-damage insurance: Optional; covers damage to your own vehicle. Accident claim: Victims or their family must file a petition before the Motor Accidents Claims Tribunal (MACT) within 6 months of the accident. Hit-and-run: Compensation from the Solatium Fund — apply at any public sector general insurer. Third-party premium rates: Fixed by IRDAI — cannot be increased arbitrarily. No-Claim Bonus (NCB): Discount on own-damage premium for claim-free years — up to 50% after 5 years. Claim rejection: Fight through Insurance Ombudsman (cioins.co.in) — free, no lawyer needed.'),
    KBEntry('ba_wallet_prepaid', 'banking_finance', 'Digital wallet and prepaid payment instrument rights', "Prepaid Payment Instruments (PPIs) — e-wallets, prepaid cards — regulated by RBI under PSS Act 2007. Types: Semi-closed (used at merchant network — e.g., Paytm, PhonePe wallet), Closed (used only with the issuer — e.g., Amazon Pay for Amazon only), Open (can be used anywhere + ATM withdrawal — e.g., prepaid bank cards). KYC: Full KYC mandatory for wallets above ■10,000/month limit. Without KYC, wallet is limited to ■10,000 balance and ■10,000/month loading. Expiry: Wallet funds cannot expire (RBI mandate). Fraud: Same 'zero liability' rules as bank accounts apply — report within 3 days for zero liability. If wallet company shuts down: RBI requires PPIs to maintain ring-fenced escrow of all wallet balances — funds are protected. Dispute: First approach wallet's in-app grievance mechanism; if unresolved in 30 days, escalate to RBI Ombudsman at cms.rbi.org.in."),
    KBEntry('ba_income_tax_tds', 'banking_finance', 'TDS, Form 16 and tax deduction rights', 'TDS (Tax Deducted at Source): Employer deducts TDS from salary based on estimated annual income. Bank deducts 10% TDS on FD interest above ■40,000/year (■50,000 for senior citizens). Form 16: Employer must give Form 16 by June 15 each year — it shows total salary and TDS deducted. Form 26AS/Annual Information Statement (AIS): Shows all TDS deducted against your PAN — view at incometaxindiaefiling.gov.in → AIS. If TDS is wrongly deducted: (1) Ask the deductor (employer/bank) to correct and file a revised TDS return. (2) If they do not, file ITR and claim the excess as refund. (3) File complaint with the Income Tax Officer. 15G/15H form: If your income is below the taxable limit, submit Form 15G (below 60) or Form 15H (senior citizens) to your bank to prevent TDS deduction on FD interest. Lower TDS certificate: If your actual tax liability is lower than TDS rate, apply for a lower deduction certificate from the Income Tax Officer.'),
    KBEntry('ba_bankruptcy_insolvency', 'banking_finance', 'Personal insolvency and debt relief options', "Insolvency and Bankruptcy Code 2016 (IBC): Personal insolvency: Individuals and partnership firms can file for insolvency before the Debt Recovery Tribunal (DRT). Fresh start process: For individuals with income below ■60,000/year, assets below ■20,000, qualifying debt below ■35,000 — can apply to DRT for discharge of debt. Insolvency resolution process: Repayment plan is negotiated with creditors under a Resolution Professional. Debt recovery: Creditors above a threshold can file to recover debt through DRT and DRAT (appellate). For honest debtors: IBC provides a 'fresh start' and 'automatic stay' on creditor actions — no harassment during the process. NPA settlement: Banks offer OTS (One Time Settlement) for NPAs — negotiate; get the settlement in writing. Section 138 Negotiable Instruments Act: Dishonour of cheque (bounced cheque) — creditor can file a criminal complaint. Accused can be imprisoned up to 2 years or fined 2× cheque amount. If cheque has bounced: Serve a statutory notice within 30 days; file complaint in court within 30 days of notice expiry."),
    KBEntry('ba_microfinance', 'banking_finance', 'Microfinance rights and SHG protections', 'Microfinance loans: Provided by NBFC-MFIs (regulated by RBI) to low-income borrowers, typically women in Self Help Groups (SHGs). RBI regulations for MFIs: (1) Annual household income cap: rural ■3 lakh, urban ■3 lakh. (2) Maximum outstanding loan: ■3 lakh per borrower. (3) Repayment cannot exceed 50% of monthly household income. (4) Minimum 24-hour cooling-off period before disbursement. (5) No security deposit from borrower. If an MFI charges above-market interest or harasses for repayment: (1) File a complaint at sachet.rbi.org.in. (2) File an FIR for criminal intimidation if there is harassment. SHG-Bank Linkage Programme: NABARD-supported — connect to your nearest rural bank or PACS. PM Vishwakarma scheme: Collateral-free loans for traditional artisans and craftspeople — pmvishwakarma.gov.in. MUDRA loans: Available to SHGs and women entrepreneurs — apply at any bank. Education & Institutional Rights 2 existing | 17 new | 19 total'),
    KBEntry('ed_law', 'education', 'Education rights: RTE, UGC, anti-ragging', 'Right to Education (RTE) Act 2009: Free and compulsory education for children aged 6–14 in government schools. Private schools must reserve 25% seats for economically weaker sections (EWS). Schools CANNOT: conduct screening tests for admission at elementary level. Withhold TC/migration certificates for non-payment of dues. Expel students (below Class 8). UGC Regulations on Ragging 2009: Ragging is a criminal offence. Punishments: suspension, expulsion, FIR, fine. Every college must have an anti-ragging committee and helpline number displayed prominently. Fee regulation: States have Fee Regulatory Committees for self-financing colleges — excess fee collection is illegal. Scholarship delays: Contact the National Scholarship Portal (scholarships.gov.in) for central government scholarships.'),
    KBEntry('ed_action', 'education', 'How to act on education-related grievances', "For Ragging: Step 1: Report to the Anti-Ragging Helpline (1800-180-5522 — toll-free, 24x7, anonymous). Step 2: File a written complaint with the college's Anti-Ragging Committee. Step 3: If the institution is unresponsive, complain to the UGC (for university) or AICTE (for technical college) online. Step 4: File an FIR at the police station — ragging can constitute assault, criminal intimidation, hurt. For fee dispute / TC withheld: Step 1: Send a written request to the Principal/Registrar with acknowledgement. Step 2: If refused, file with the State Fee Regulatory Committee (for private colleges) or the Education Department. Step 3: File a consumer complaint at e-daakhil.nic.in — withholding TC/certificate is a service deficiency. For scholarship: Check status on scholarships.gov.in; contact the scheme's helpline; file RTI for scheme information."),
    KBEntry('ed_rte_detailed', 'education', 'RTE Act — detailed rights for children and parents', 'Right to Education Act 2009 (RTE): (1) Free and compulsory education for ALL children aged 6–14 in a neighbourhood school. (2) 25% reservation for EWS/disadvantaged children in private unaided schools — school must admit and provide free education. (3) No capitation fee or screening for admission at the elementary level — violation is an offence. (4) Age-appropriate admission: Children must be admitted in the class appropriate for their age even mid-year. (5) Schools must have: teachers in prescribed pupil-teacher ratio, building with toilets, clean drinking water, playground. Complaints for RTE violation: File with the District Education Officer (DEO) or Block Education Officer (BEO). Approach the State Commission for Protection of Child Rights (SCPCR). NCPCR: ncpcr.gov.in — national-level escalation. Prohibition on punishment: No child shall be subjected to physical punishment or mental harassment — it is an offence (Section 17 RTE).'),
    KBEntry('ed_higher_edu_rights', 'education', 'Higher education rights — UGC, AICTE and university rules', "UGC (University Grants Commission) regulates central/deemed universities and grants. AICTE (All India Council for Technical Education) regulates engineering, pharmacy, architecture, MBA institutions. Your rights in higher education: (1) Institutions must display fees, affiliations and recognition on notice boards and website. (2) Fee refund on withdrawal: UGC Refund Policy — refund of fees if withdrawal is before the admission deadline (85% after the deadline, reducing in steps). (3) Examination re-evaluation: Most universities allow application for re-evaluation of answer sheets — apply within prescribed time. (4) Caste discrimination in admission: File complaint with the institution's anti-discrimination cell or the UGC. (5) Fake universities: UGC publishes a list of fake/de-recognised universities — verify at ugc.ac.in before admission. For grievances: National Student Grievance Portal (grievance.ugc.ac.in) and AICTE portal at grievance.aicte-india.org."),
    KBEntry('ed_scholarship_schemes', 'education', 'Scholarships, fellowships and financial aid rights', 'Central government scholarship schemes (National Scholarship Portal — scholarships.gov.in): (1) Post Matric Scholarship for SC students: funded by central government, managed by states. (2) Post Matric Scholarship for OBC and EBC students. (3) National Merit-cum-Means Scholarship (NMMS): for meritorious students from poor families in Class 9–12. (4) Central Sector Scheme of Scholarships for College students (CSSS). (5) Maulana Azad National Fellowship for Minority students. (6) National Fellowship and Scholarship for Higher Education of ST students. UGC research fellowships: JRF (Junior Research Fellowship) via NET/CSIR. State scholarships: State social welfare departments disburse state government scholarships. If scholarship is delayed: File an RTI or complaint on NSP portal. Minority scholarship: PM Yasasvi scheme for OBC/EBC/DNT students at yasasvi.nta.ac.in.'),
    KBEntry('ed_entrance_exam', 'education', 'Entrance exam rights and NTA/competitive exam grievances', "NTA (National Testing Agency) conducts JEE, NEET, CUET, UGC NET and other national exams. Candidate rights: (1) Free and fair examination — any allegation of paper leak or impersonation can be reported to NTA and CBI. (2) Right to see your answer sheet: NTA must display provisional answer keys and invite objections with a fee. (3) Incorrect answer key: File an objection within the NTA's objection window — expert panel reviews. (4) Admit card not received: Contact NTA helpline — 011-69227700 or exams.nta.ac.in. (5) Disability accommodation: PwD candidates are entitled to scribe, extra time, and accessible centres — apply at the time of registration. NEET (medical entrance): Seat matrix and admission via Medical Counselling Committee (MCC) — mcc.nic.in. JEE Advanced: IIT admission — JoSAA counselling portal. State entrances: State-specific boards (e.g., MH-CET, KEAM) — governed by respective state boards."),
    KBEntry('ed_anti_ragging', 'education', 'Anti-ragging law — detailed procedure and protections', 'Ragging is defined broadly under UGC Anti-Ragging Regulations 2009 to include: teasing, bullying, humiliating, forcing consumption of alcohol/drugs, sexual harassment, physical assault of juniors. Zero tolerance: Any proven act of ragging leads to: expulsion from institution; FIR by the institution; imprisonment up to 3 years. Anti-Ragging Helpline: 1800-180-5522 (toll-free, 24x7, anonymous complaints accepted). Online FIR: antiragging.in portal. Institutions must: have a visible anti-ragging committee, display helpline numbers, take an affidavit from every student and parent, conduct awareness sessions. Failure to act by institution: Complain to UGC (for universities), AICTE (for technical colleges), MCI (for medical colleges). The institution itself can be fined or have its affiliation withdrawn. Survivor support: iCall (9152987821) provides free counselling for ragging survivors.'),
    KBEntry('ed_disability_education', 'education', 'Rights of students with disabilities in education', "Rights of Persons with Disabilities Act 2016 (RPWD): (1) Every educational institution (government and government-aided) must provide inclusive education for students with benchmark disabilities. (2) Students with specified disabilities are entitled to: extra time in exams (typically 25–50%), scribe/reader, accessible infrastructure, exemption from certain subjects (on application). (3) 5% reservation in admissions in government educational institutions (higher education). (4) Schools cannot refuse admission or expel a student solely on grounds of disability. CBSE/State Board accommodations: Apply to the board through your school — disability certificate from a government hospital required. UGC guidelines for PwD students in universities: Cover compensatory time, alternate format materials, accessible hostels. Complaint: Approach the institution's Disability Cell, then the State Commissioner for Persons with Disabilities. NIEPID (National Institute for Empowerment of Persons with Intellectual Disabilities): technical support for institutions."),
    KBEntry('ed_female_student', 'education', 'Rights of female students — safety, hostel and POSH', 'POSH Act applies to educational institutions: Sexual harassment of female students by teachers, staff or other students can be reported to the Internal Complaints Committee (ICC) of the institution. Hostel safety: UGC regulations require universities to have well-lit, secure hostels with CCTV and wardens. Dress code and moral policing by institutions: Arbitrary dress codes that target women specifically can be challenged as discriminatory. Night curfew for female students (hostels): Any curfew more restrictive for women than for men, without safety justification, can be challenged as discriminatory (High Court rulings in Gujarat and other states). Pregnancy / maternity leave for female students: UGC regulations allow maternity leave and late submission of work. Institutions cannot expel a student for being pregnant. Menstrual leave: Some state governments and institutions have policies; no central law as yet. Safety on campus: Internal Safety Committee + complaint mechanism required under UGC regulations. Complaint: UGC SHe-Box for university students; institution ICC; local police for serious incidents.'),
    KBEntry('ed_minority_institution', 'education', 'Minority educational institutions — rights and protections', "Article 30: Religious and linguistic minorities have the right to establish and administer educational institutions. Minority institution status: Granted by the National Commission for Minority Educational Institutions (NCMEI) — apply at ncmei.gov.in. Benefits: Exempted from reservations in admissions for minority community seats. Right to appoint teachers of their choice. Aided minority institutions: Can receive government aid but must comply with prescribed service conditions for teachers. Unaided minority institutions: Almost complete autonomy in admissions and administration. RTE Act applicability: Supreme Court (CISCE case) held that RTE's 25% EWS reservation does NOT apply to unaided minority schools. NCMEI: Dispute resolution for minority status claims and protection of minority institution rights. For complaints about minority institutions: State Minority Commission or NCMEI."),
    KBEntry('ed_online_education', 'education', 'Online education rights, degree validity and SWAYAM', 'UGC (Open and Distance Learning) Regulations 2020: Distance and online degrees from UGC-recognised institutions are valid for government jobs and higher studies. SWAYAM platform (swayam.gov.in): Government online courses — SWAYAM credits can be transferred to university grades (up to 40% of total credits in many universities). Fake online degrees: Verify university recognition at ugc.ac.in before enrolling. MOOCs and micro-credentials: Not equivalent to degree programmes — verify employer recognition before spending money. EdTech consumer rights: If a paid online course is not delivered as promised: file a consumer complaint at e-daakhil.nic.in. Student data privacy: EdTech companies must comply with IT Act and forthcoming DPDP Act for student data. Online proctored exams: Students have the right to know what proctoring software is being used and what data is collected. IGNOU and state open universities: Provide valid distance education degrees — check recognition for your specific programme.'),
    KBEntry('ed_fee_refund', 'education', 'Fee refund and cancellation rights in educational institutions', 'UGC Fee Refund Policy: Applicable to all UGC-regulated institutions. (1) If student withdraws before the commencement of the academic session: 100% refund of fees (deducting processing fee of maximum ■1,000). (2) Up to 15 days after commencement: 80% refund. (3) 15–30 days after commencement: 50% refund. (4) 30–45 days after commencement: 25% refund. (5) More than 45 days: No refund. This policy applies to all types of fees (tuition, hostel, etc.) except caution deposit. If institution refuses to refund as per UGC policy: File a grievance at grievance.ugc.ac.in. TC and migration certificate: Institution CANNOT withhold TC as a condition for fee payment — it must be issued within the prescribed time. For AICTE-regulated institutions: Same policy applies — grievance at grievance.aicte-india.org. Consumer complaint: Withholding TC or refusing fee refund as per policy is a consumer deficiency — file at e-daakhil.nic.in.'),
    KBEntry('ed_teacher_rights', 'education', 'Teacher rights and employment protections in education', 'School teacher rights (government): Governed by state service rules; disputes before the Administrative Tribunal or High Court. University teacher rights: UGC Regulations on minimum qualifications for appointment; NET/PhD mandatory for most posts. Probation period: Usually 2 years for government school/university teachers — regularisation after satisfactory service. Academic freedom: Teachers have the right to academic expression — cannot be penalised for views expressed in an academic context. POSH Act: Applies to teachers; each educational institution must have an ICC. Contractual/adhoc teachers: Entitled to minimum wages; courts have held that long-serving ad-hoc teachers have a right to regularisation in many state cases. Private school teachers: Minimum salary as per state education department order — many states have prescribed pay scales. Wrongful termination: Approach the Labour Court or industrial dispute mechanism; also National Commission for Scheduled Castes if caste-based discrimination.'),
    KBEntry('ed_student_loans', 'education', 'Education loan rights and schemes', "Education loan: IBA Model Education Loan Scheme (extended to Vidya Lakshmi portal — vidyalakshmi.co.in): Apply to multiple banks through a single portal. Loan up to ■7.5 lakh: No collateral required (under IBA scheme). Loan above ■7.5 lakh: Collateral required; moratorium period = course duration + 1 year. Repayment: Begins 1 year after course completion (or 6 months after getting a job, whichever is earlier). Interest subvention: Full interest subvention for economically weaker sections (annual family income below ■4.5 lakh) during the moratorium — under PM-VIDYA LAKSHMI scheme. Credit Guarantee Fund: Allows banks to give loans without collateral for amounts up to ■7.5 lakh — NCGTC manages this. Bank cannot refuse education loan for a recognized institution: report refusal to the bank's grievance officer or RBI. Tax benefit: Interest paid on education loan is deductible from income for 8 years (Section 80E IT Act). Repo rate linked education loans: Floating rate loans reset periodically — ask your bank for the reset schedule."),
    KBEntry('ed_school_safety', 'education', 'School safety — transport, infrastructure and POCSO', 'CBSE/State Board guidelines on school safety: Schools must conduct periodic safety audits, fire drills, and maintain fire safety equipment. School bus safety (Supreme Court guidelines + Motor Vehicles Act): (1) Bus driver must have PSV (Public Service Vehicle) badge. (2) Attendant mandatory for primary school children. (3) Bus must have first aid kit, fire extinguisher, emergency exits. (4) Speed governor mandatory — maximum 40 km/h with children on board. POCSO compliance: Every school must have a child protection policy. Staff must be trained. Any act of sexual misconduct by a teacher or staff member must be reported to police immediately — failure to report is an offence. Corporal punishment: Prohibited under RTE Act — file complaint with DEO or BEO. Infrastructure: Safe drinking water, functional toilets (separate for girls), ramps for PwD — mandatory under RTE. Complaint for school safety violation: DEO → State Director of Education → NCPCR.'),
    KBEntry('ed_foreign_study', 'education', 'Rights of Indian students studying abroad', "Before enrolling: (1) Verify university recognition — WES (World Education Services) evaluation helps for Canada/US. QS rankings, NARIC for UK, DAAD for Germany. (2) Check if the degree is valid for government jobs in India — AIU (aiu.ac.in) evaluates foreign degrees. (3) Avoid 'diploma mills' — fake universities with purchased degrees are prevalent online. Student visa rights abroad: Consult the Indian Embassy/High Commission if you face exploitation by the university or employer. If stranded abroad: Call MEA's 24x7 helpline: +91-11-23012113, or the nearest Indian Embassy. FRRO registration: Foreign students studying in India must register with the Foreigners Regional Registration Office (FRRO) within 14 days of arrival at frro.gov.in. Education loans for abroad: Same Vidya Lakshmi portal; collateral usually required; higher loan amounts available. FEMA: Foreign exchange for education is allowed up to USD 2.5 lakh per year without RBI permission under the Liberalised Remittance Scheme (LRS)."),
    KBEntry('ed_vocational_skills', 'education', 'Vocational training and skill development rights', 'PMKVY (Pradhan Mantri Kaushal Vikas Yojana): Free skill training + certification + placement support for youth. Register at pmkvyofficial.org or through your nearest training centre. Skill India portal: skillindia.gov.in — find government-funded courses near you. National Skill Qualification Framework (NSQF): All vocational qualifications are aligned to NSQF levels (1–10), allowing vertical progression. ITI (Industrial Training Institute) rights: Free education for SC/ST students in government ITIs in many states. NCVT/SCVT certificates are nationally/state recognised. If training centre is fake/disappears with fees: File consumer complaint and FIR. National Apprenticeship Promotion Scheme (NAPS): Stipend support to employers who take on apprentices. Recognition of Prior Learning (RPL): Workers with informal skills can get formal certification — check PMKVY portal. e-Shram portal (eshram.gov.in): Unorganised workers can register and access government scheme benefits.'),
    KBEntry('ed_examination_rights', 'education', 'Examination rights — re-evaluation, improvement and appeals', "Re-evaluation / re-checking: Most boards/universities allow application for re-checking (recount of marks) and re-evaluation (reassessment by another examiner). Apply within the prescribed window (usually 15–30 days of result). Fee is charged; refunded if marks increase significantly in many boards. Right to see answer sheet: Several High Courts have ruled that students have a right to see their evaluated answer sheets under RTI — file RTI with the board if re-evaluation is denied. Grace marks: Some boards have grace mark policies for border-line candidates — ask the board for the applicable policy. Compartment / improvement: Most boards allow students who fail in 1–2 subjects to appear in a compartment exam. CBSE improvement exam: Students who clear Class 12 can appear for improvement in up to 5 subjects in the next year. Unfair means (UFM) charge: If accused, you have the right to a personal hearing before the UFM committee — submit your defence in writing. Board exam grievance: File at the board's official portal; escalate to the state education secretary if unresolved."),
    KBEntry('ed_library_info_access', 'education', 'Right to information in libraries and academic resources', 'National Digital Library of India (NDLI — ndl.gov.in): Free access to millions of academic texts, books, journals and educational resources for registered students. INFLIBNET (inflibnet.ac.in): Consortium of university libraries — provides access to thousands of research journals for students/researchers at member universities. Open Access: Researchers funded by government grants (DST, DBT, ICMR) must make publications open access — institutions must have open access repositories. RTI for academic information: University examination records, marking schemes, number of scripts evaluated — accessible under RTI. Copyright and study materials: Fair dealing under Section 52 Copyright Act allows students to make copies of portions of works for private study. Academic plagiarism: UGC Regulations 2018 (Promotion of Academic Integrity and Prevention of Plagiarism) — acceptable plagiarism threshold: 10% or less. Above 40% similarity: paper retracted, degree may be cancelled. Research data access: Government-funded research data must be made publicly available — check individual funding agency policies. Family & Personal Law 4 existing | 15 new | 19 total'),
    KBEntry('fa_law', 'family_personal', 'Domestic violence and family law protections', "Protection of Women from Domestic Violence Act 2005 (PWDVA): Covers: current or former spouses, live-in partners, relatives. Types of abuse covered: Physical, emotional, verbal, sexual, AND economic abuse (denying money, throwing you out of the shared home). Legal remedies available through a Magistrate: (1) Protection Order: abuser is prohibited from contacting, entering the home, etc. (2) Residence Order: you have the RIGHT to stay in the shared household even if you don't own it. (3) Monetary Relief: compensation for injuries, medical expenses, loss of earnings. (4) Custody Order: temporary custody of your children. Dowry Prohibition Act 1961: Giving or taking dowry is a criminal offence. Section 498A IPC (now BNS) punishes cruelty by husband/in-laws."),
    KBEntry('fa_action', 'family_personal', 'How to get help in domestic violence or family crisis', "Immediate safety: Call 112 (Police emergency) or 181 (Women's Helpline — 24x7, connects to police/shelter). Go to the nearest police station and file an FIR for assault, criminal intimidation, or under PWDVA/Section 498A BNS. Approaching the Magistrate directly (no police needed): You can file an application under PWDVA directly in the Magistrate's court — no lawyer required, but helpful. The Magistrate can pass an emergency protection order on the SAME DAY. Free resources: (1) Protection Officer: every district has one — visit the District Women and Child Development office. (2) One Stop Centres (Sakhi Centres): shelter, medical, legal and counselling support — free. (3) Swadhar Greh: shelter homes for women in difficult circumstances. (4) Legal Aid: NALSA helpline 15100 — free legal representation for women in domestic violence cases. Evidence to preserve: Photos of injuries, medical reports, threat messages/call recordings, witnesses."),
    KBEntry('fa_divorce_maintenance', 'family_personal', 'Divorce, maintenance and child custody', 'Maintenance rights: Section 125 CrPC (now BNSS Section 144): Any wife, minor child, or elderly parent can claim maintenance from a husband/father/son. Court can grant interim maintenance quickly (often within 60–90 days). Special Marriage Act applies to inter-religion marriages. Hindu Marriage Act, Muslim Personal Law, Christian Marriage Act apply to respective communities. Divorce types: Mutual consent divorce is fastest (6 months cooling period, waiveable in some cases). Contested divorce takes longer — 1–5 years typically. Child custody: Courts prioritise the best interest of the child. Mothers typically get custody of young children; courts may arrange joint custody. Child can express preference to the court at age 9+. NRI divorce: If your spouse is abroad, Special Marriage Act courts in India have jurisdiction if marriage was solemnised in India. International child abduction: Approach the Ministry of Women and Child Development.'),
    KBEntry('fa_senior_citizen', 'family_personal', 'Senior citizen rights and elder abuse', 'Maintenance and Welfare of Parents and Senior Citizens Act 2007 (amended 2019): Children and grandchildren are legally OBLIGATED to maintain parents and grandparents who cannot support themselves. If neglected: Senior citizen can file an application before the Maintenance Tribunal (usually SDM office) — simple process, no court. Tribunal can order children to pay up to ■10,000/month maintenance. Property gifted to children: If children neglect the senior after receiving property, the gift can be CANCELLED by the Tribunal. Senior Citizen Helpline: 14567 (Elderline — 24x7). For physical abuse/abandonment: File a police complaint. Contact Agewell Foundation, HelpAge India for counselling and legal support. Old age homes: Senior citizens can approach the District Social Welfare Officer for government shelter.'),
    KBEntry('fa_child_custody', 'family_personal', 'Child custody — interim, permanent and international', "Child custody is governed by the personal law applicable to the parents (Hindu Minority and Guardianship Act, Guardians and Wards Act 1890 — applicable to all). Courts' primary consideration: Best interest of the child. Interim custody: Applied for at the start of proceedings — Family Court may grant interim custody within weeks. Types: Sole custody, joint custody, joint legal custody. Mother's custody presumption: Courts typically grant custody of young children to the mother — but this is not absolute. Father can get custody if it is in the child's best interest. Child's preference: Children above 9–13 years of age may express preference to the Family Court. Visitation rights: The non-custodial parent has a right to regular access/visitation unless the court restricts it. International custody (Hague Convention): India is not a signatory — if a child is taken abroad without consent, approach the Ministry of Women and Child Development and the destination country's courts. Parental alienation: A court can hold a parent in contempt for denying the other parent their visitation rights."),
    KBEntry('fa_maintenance_detail', 'family_personal', 'Maintenance and alimony — how to claim', "Section 144 BNSS (formerly Section 125 CrPC): Any wife, minor child, or parent unable to maintain themselves can claim maintenance from husband/father/son. Interim maintenance: Courts usually grant within 60–90 days of application. Calculation: Court considers income of both parties, standard of living, responsibilities — no fixed formula. Non-payment of maintenance: File a contempt petition before the Family Court — court can issue a warrant of attachment of salary/assets. Hindu law (HAMA 1956): Wife, children and parents of a Hindu male have maintenance rights. Muslim law: Mehr (dowry) is the wife's right on marriage/divorce; iddat maintenance for 3 months after talaq. Shah Bano judgement (1985) and Muslim Women (Protection of Rights on Marriage) Act 2019 give additional rights. Christian and Parsi divorce maintenance: Indian Divorce Act and Parsi Marriage and Divorce Act apply respectively. Arrears: Court can recover accumulated unpaid maintenance from the husband's assets."),
    KBEntry('fa_adoption', 'family_personal', 'Adoption laws and CARA registration', 'Hindu Adoptions and Maintenance Act 1956 (HAMA): Governs adoption among Hindus, Sikhs, Jains, Buddhists. Adoptive parents must be Hindu. Juvenile Justice Act 2015 (secular): Governs adoption by all religions and foreign adoption — through CARA (Central Adoption Resource Authority). CARA (cara.nic.in): Nodal body for child adoption. All domestic adoption must now go through CARA (even for Hindus, as per Supreme Court 2022). Eligibility: Married couple (2 years of stable marriage), single women, single men (not for girl child) — age and income criteria apply. Process: Register on CARA, home study by licensed agency, wait for child referral, court order, adoption deed. Inter-country adoption: More restrictions post-Hague Convention — courts require CARA clearance. Illegal adoption (informal): Not legally valid — the child has no inheritance rights without legal adoption. Foster care: Child is not legally adopted — registered under JJ Act with Child Welfare Committee.'),
    KBEntry('fa_muslim_personal_law', 'family_personal', 'Muslim personal law — marriage, divorce and rights', "Muslim personal law (Shariat Application Act 1937) governs: marriage, divorce, maintenance, guardianship, inheritance for Muslims. Nikahnama (marriage contract): Should specify Mehr amount; additional conditions can be added (e.g., right of wife to divorce, prohibition on second marriage). Talaq: Triple talaq in one sitting declared unconstitutional (Shayara Bano v. Union of India, 2017). Muslim Women (Protection of Rights on Marriage) Act 2019: Instant triple talaq is a criminal offence — punishable with up to 3 years imprisonment. Khula: Wife can seek divorce with return of Mehr — through a Family Court or by mutual agreement. Mehr: Wife's absolute right — cannot be waived without proper consent. Second marriage (polygamy): Currently legal under Muslim personal law but challenged in courts and ongoing legislative debate. Inheritance: Muslim inheritance is governed by Muslim personal law — daughters get half the share of sons. Uniform Civil Code: Being debated nationally; only Goa has a uniform civil code as of now."),
    KBEntry('fa_hindu_marriage', 'family_personal', 'Hindu Marriage Act — rights in marriage and divorce', "Hindu Marriage Act 1955 (HMA): Applies to Hindus, Sikhs, Jains, Buddhists. Conditions for valid marriage: Monogamy, minimum age (18 for women, 21 for men — minimum age for men under discussion), neither party a lunatic or idiot, prohibited degrees of relationship. Child marriage: Prohibited under Prohibition of Child Marriage Act 2006 — punishable with imprisonment; marriages may be void or voidable. Nullity of marriage: Can be declared void (if bigamous, prohibited degrees) or voidable (impotence, unsound mind at time of marriage). Grounds for divorce under HMA: Adultery, cruelty, desertion (2 years), conversion to another religion, mental illness, incurable disease, renouncement, civil death. Mutual consent divorce: 6-month waiting period (can be waived by the court in some circumstances). Judicial separation: Alternative to divorce — parties remain married but don't have to cohabit. Restitution of conjugal rights: Court order to cohabit — cannot be enforced by compulsion."),
    KBEntry('fa_domestic_violence_men', 'family_personal', 'Domestic violence against men and gender-neutral approach', "Current law (PWDVA 2005): Only women can file as aggrieved persons; men can be respondents. There is no equivalent law protecting men. BNS remedies available to men: (1) Section 296: Criminal intimidation (threats). (2) Section 115: Causing hurt. (3) Section 86 complaint (against wife for cruelty by wife's relatives in some interpretations — not the main provision). False Section 498A cases: If a man is falsely accused of cruelty by wife/in-laws, he can: apply for anticipatory bail immediately; file a complaint for malicious prosecution/perjury if proved false. Maintenance rights of separated fathers: Fathers can claim custody; maintenance is usually paid by the father to wife/child, but courts consider income of both. iCall helpline (9152987821): Counselling for men in domestic distress. Save Indian Family Foundation: NGO providing support to men in matrimonial disputes. Men in genuine distress with threatening/violent spouse: File an FIR for assault/criminal intimidation under BNS; seek anticipatory bail."),
    KBEntry('fa_live_in_rights', 'family_personal', 'Live-in relationship rights in India', "Supreme Court recognition: A long-term live-in relationship that has the 'hue of a marriage' is treated as a valid marriage for certain purposes — domestic violence protection, maintenance. Domestic Violence Act 2005 applies to live-in partners: A woman in a live-in relationship can file for protection order, residence order, and maintenance under PWDVA. Child from live-in relationship: Legitimate — has the right to inheritance from both parents. Property rights: No automatic property rights — they depend on what was agreed or contributed. Section 144 BNSS maintenance: Live-in partner can claim maintenance if the relationship has lasted for a significant period and has the character of a marriage. Break-up of live-in: No legal separation procedure required — but the woman can claim her belongings and seek protection order if threatened. Registration: No registration mechanism exists for live-in relationships in India currently."),
    KBEntry('fa_matrimonial_fraud', 'family_personal', 'Matrimonial fraud and NRI marriage fraud', 'NRI marriage fraud: Indian citizen marries and spouse goes abroad and abandons them. Very common complaint. Legal remedies: (1) File an FIR for cheating (BNS Section 316) and cruelty (BNS Section 86). (2) Apply for a Look Out Circular (LOC) through the police to prevent the spouse from leaving India. (3) File for maintenance before the Family Court. (4) For passport impoundment of the non-cooperative NRI spouse: Apply to the Regional Passport Office/MEA. Ministry of External Affairs: Indian Women Abroad Cell — +91-11-23013139 / mea-hoc@mea.gov.in. Bigamy: If the person was already married and married you without disclosing this, file FIR for bigamy (BNS Section 82) + cheating. Matrimonial website fraud: If a profile was fabricated on a matrimonial site — report to cybercrime.gov.in and the platform. Dowry recovery: File a civil suit or complaint with the Dowry Prohibition Officer.'),
    KBEntry('fa_child_support', 'family_personal', 'Child support, maintenance and custody enforcement', "Maintenance for minor children: Section 144 BNSS / Section 26 HMA — either parent can be ordered to pay maintenance for the child. Father's obligation: Even if the father has no direct income (agricultural land, self-employment), the court imputes income to determine maintenance. Education expenses: Courts can order the non-custodial parent to pay school fees, medical expenses, etc. in addition to maintenance. Enforcement: If the maintenance order is not complied with, file an execution petition in the Family Court — court can issue a warrant for attachment of salary or property. Modification: Maintenance orders can be revised if there is a change in circumstances (increase in payer's income, change in child's needs). International enforcement: If the paying parent is abroad, enforcement is complex — consult a lawyer; India has mutual legal assistance treaties with some countries. Working parent contributing more: Courts may order the mother (if earning) to also contribute — child's best interest is paramount. Grandparents: If both parents are unable to maintain the child, grandparents may be ordered to pay under Guardians and Wards Act."),
    KBEntry('fa_domestic_worker_rights', 'family_personal', 'Domestic worker and household staff rights', "No central law specifically for domestic workers — but multiple protections exist: (1) Minimum Wages Act: Domestic workers are included in minimum wage schedules in most states (check your state's schedule). (2) POSH Act: Applies to domestic workers — they can file with the Local Complaints Committee (LCC) of the district. (3) NDWM (National Domestic Workers Movement): NGO providing legal aid and advocacy. (4) e-Shram portal: Domestic workers can register and get ■2 lakh accident insurance. Rights in practice: Right to weekly rest day; right to notice before termination; right to wages for worked days including notice period. Child domestic workers: Employing children below 14 as domestic workers is ILLEGAL — report to CHILDLINE 1098. Sexual harassment of domestic workers: File with LCC (not ICC — there is no ICC in private households); call 181 Women Helpline. Exploitation/abuse: File an FIR; contact the nearest Labour Inspector; approach the District Social Welfare Officer."),
    KBEntry('fa_intercaste_interreligion', 'family_personal', 'Inter-caste and inter-religion marriage rights', "Special Marriage Act 1954: Allows any two persons of any religion or caste to marry. Gives the couple a uniform civil code for personal matters after marriage (Hindu Succession Act applies to property rights by default if both are Hindus). Registration: Under Special Marriage Act, 30-day notice must be given to the Marriage Officer — public notice can expose couples to harassment. Karnataka High Court and other courts have said notice provisions can be waived for safety. Hindu Marriage Act: Inter-caste Hindu marriages are fully valid and LEGAL — untouchability is prohibited under Article 17. Honour-based violence: If family threatens or harms a couple for an inter-caste/inter-religion marriage: (1) File an FIR immediately. (2) Seek police protection — Shaheen Abdullah case guidelines (2011) — Supreme Court mandated police protection for inter-caste couples. (3) File a Habeas Corpus petition in the High Court if one partner is confined by family. Special Cell: Many states have 'love jihad' related registrations — these can be challenged in court. NGO support: Dhanak (dhanakindia.org) provides shelter and support to inter-caste/religion couples."),
    KBEntry('fa_surrogacy', 'family_personal', 'Surrogacy rights and regulations in India', 'Surrogacy (Regulation) Act 2021: (1) Commercial surrogacy is BANNED. Only altruistic surrogacy (by a close relative) is permitted. (2) Eligible intending parents: Married Indian couple (woman 25–50, man 26–55), widowed/divorced woman (35–45). (3) Surrogate must be: a close relative, married, has her own child, 25–35 years, only once in her lifetime. (4) Approval from State Board and Appropriate Authority is mandatory. Foreign nationals cannot commission surrogacy in India. If the surrogate mother wishes to withdraw: She can withdraw consent before embryo implantation. Child born of surrogacy: The intending couple is the legal parent — surrogate has no parental rights after birth. National Assisted Reproductive Technology and Surrogacy Board (NATSB): regulates all ART clinics and surrogacy procedures. Violation of the Act: Criminal offence — fine up to ■10 lakh and/or imprisonment.'),
    KBEntry('fa_consumer_family', 'family_personal', 'Consumer rights in family services — weddings, events', "Wedding and event services: Caterers, decorators, venues, photographers, wedding planners are all service providers under the Consumer Protection Act 2019. Common disputes: Venue cancels at the last minute; caterer delivers substandard food; photographer loses photos; decorator does not deliver as promised. Your rights: Right to compensation for mental agony + actual financial loss + cost of substitute service. Action: (1) Document everything: contract, photos, communication. (2) Send a written complaint to the service provider. (3) File a consumer complaint at e-daakhil.nic.in — District Consumer Commission handles claims up to ■50 lakh. Wedding trousseau (shopping): Returns and exchanges are at the merchant's discretion unless goods are defective. Defective goods: consumer right to refund/replacement. Jewellery: BIS Hallmarking mandatory — demand HUID certificate."),
    KBEntry('fa_inheritance_women', 'family_personal', "Women's inheritance rights under Hindu law", "Hindu Succession Act 1956, Section 6 (amended 2005): Daughters have equal coparcenary rights in ancestral property by birth — same as sons. Key Supreme Court ruling (Vineeta Sharma v. Rakesh Sharma, 2020): Daughters' rights apply even if the father died before the 2005 amendment came into force. Self-acquired property: Father can will it to anyone — daughters do not have automatic rights. Mother's property: Passes equally to all children (both sons and daughters). Married daughter: Marriage does not diminish the daughter's inheritance rights. If denied inheritance: (1) File for partition of ancestral property in Civil Court. (2) Send a legal notice to co-heirs. (3) Register your name in revenue records — approach tehsildar with supporting documents. Widow's rights: Hindu widow gets a share equal to each son in the husband's estate; she retains her stridhan (jewellery, gifts) absolutely. NALSA women's legal aid: Call 15100 for free legal advice on inheritance."),
    KBEntry('fa_elderly_property', 'family_personal', 'Protecting property rights of elderly parents', 'Maintenance and Welfare of Parents and Senior Citizens Act 2007 (amended 2019): If a senior citizen transfers property to children with the expectation of maintenance and the children neglect them after the transfer: The transfer can be CANCELLED by the Maintenance Tribunal. Evidence required: Proof of the condition of maintenance attached to the gift/transfer, and proof of neglect. Revocation is handled by the Maintenance Tribunal (usually under the Sub-Divisional Magistrate). How to protect in advance: (1) Retain a life interest clause in the gift deed (you can live in the property till your death). (2) Include a condition of monthly maintenance in the gift deed. (3) Add a reversion clause (property reverts if conditions are breached). Senior citizen helpline: Elderline 14567. Property in a trust: Consider a family trust under Indian Trusts Act for complex succession situations. Legal aid for seniors: Free under Legal Services Authorities Act — NALSA 15100. Public Administration & Governance 5 existing | 14 new | 19 total'),
    KBEntry('go_law', 'governance_admin', 'RTI Act and anti-corruption framework', 'Right to Information Act 2005: Any citizen can request information from ANY public authority. Time limit: Public authority must respond within 30 days (48 hours for life/liberty matters). Fees: ■10 application fee for central government (free for BPL applicants). Many states have similar fees. What you can ask: Any document, record, data, sample, circular, order, memo, or email held by a public authority. What you CANNOT ask: Cabinet deliberations, defence/security matters, personal information with no public interest. If refused unjustly: First Appeal: to the First Appellate Authority within 30 days of refusal. Second Appeal: to the Central/State Information Commission within 90 days. Prevention of Corruption Act 1988 (amended 2018): A public servant demanding a bribe commits a cognizable offence. Reporting a bribe you were asked to pay is protected — the bribe-payer is NOT automatically an offender if they report it.'),
    KBEntry('go_action', 'governance_admin', 'RTI and corruption: how to file and escalate', "Filing an RTI: Step 1: Write to the Public Information Officer (PIO) of the relevant department. Step 2: Central government RTI: file online at rtionline.gov.in (pay ■10 by net banking/card). Step 3: State governments: most have online portals or accept RTI by registered post. Step 4: If no response in 30 days — First Appeal to the Appellate Authority within 30 days. Step 5: Second Appeal to Information Commission (online or by post) within 90 days of First Appeal reply. Reporting a bribe/corruption: (1) Central Vigilance Commission (CVC): complaint at cvc.gov.in or by post. (2) Lokpal of India: complaint at lokpal.nic.in for corruption by Group A/B central government officers. (3) State Lokayukta: for state government officers. (4) Anti-Corruption Bureau of your state: approach directly or file FIR. (5) Chief Minister's helpline: most states have a direct helpline (e.g., CM Helpline 1076 in many states). Whistleblower protection: The Whistle Blowers Protection Act 2014 protects those who expose corruption."),
    KBEntry('go_police_complaint', 'governance_admin', 'Police misconduct and forced FIR refusal', "If police REFUSE to register an FIR for a cognizable offence — this is illegal (S.154 BNSS). Action options: (1) Send a written complaint to the Superintendent of Police (SP) by registered post. (2) File a complaint before the Executive Magistrate (Magistrate can direct police to investigate — S.175 BNSS). (3) File an online complaint on the state police's website/CM Grievance portal. (4) Send a complaint to the State Human Rights Commission (SHRC) for rights violations by police. If police demand bribe to register FIR or during investigation: (1) Note down the officer's name, badge number, date/time. (2) Report to the Anti-Corruption Bureau. (3) File a complaint with NHRC or SHRC. For police brutality/custodial torture: This is a fundamental rights violation. File with NHRC (nhrc.nic.in) and the Magistrate. Custodial death complaints must be sent to NHRC within 24 hours by the officer in charge."),
    KBEntry('go_documents', 'governance_admin', 'Government documents: Aadhaar, PAN, passport, caste certificate', "Aadhaar: Update/correction at uidai.gov.in or Aadhaar Seva Kendra. Helpline: 1947. PAN: Apply/update at incometaxindiaefiling.gov.in or NSDL. Helpline: 020-27218080. Passport: Apply at passportindia.gov.in. Tatkaal service for urgent needs. Helpline: 1800-258-1800. Voter ID: Enroll/update at voterportal.eci.gov.in. Helpline: 1950. Caste certificate: Apply at your district collectorate / Tehsil office. If delayed beyond normal time: (1) File an RTI asking for the status of your application. (2) File a complaint on the CM's grievance portal. (3) Report to the District Magistrate's office for administrative action. If official demands bribe for certificate: Report to Anti-Corruption Bureau. Online services: Most documents now have online application/tracking — always use official government (.gov.in) websites only."),
    KBEntry('health_rights', 'governance_admin', 'Patient rights and medical negligence', "Patient rights in India: (1) Right to be told your diagnosis, treatment options, risks, and alternatives in simple language. (2) Right to informed consent before any procedure. (3) Right to a second opinion. (4) Right to see and get copies of all medical records. (5) Right to emergency treatment: A hospital (government or private) CANNOT refuse emergency treatment even if you cannot pay — this is a Constitutional right (Article 21). Medical negligence: If a doctor/hospital causes harm due to negligence: (1) File a consumer complaint (doctors are 'service providers' under the Consumer Protection Act). (2) File a complaint with the State Medical Council for professional misconduct. (3) File an FIR for criminal negligence (BNS) if severe harm or death results. Government hospital complaints: Write to the Medical Superintendent → District Chief Medical Officer → State Health Secretary. For lack of medicines in government hospitals: RTI to the health department, complaint to CMO."),
    KBEntry('go_rti_detailed', 'governance_admin', 'RTI — detailed filing guide with exemptions', 'Filing an RTI online (Central Government): rtionline.gov.in. Fee: ■10 by BHIM/net banking. Filing RTI offline: Write an application in English or Hindi (or state official language) addressed to the Public Information Officer (PIO). Pay ■10 by postal order/DD or cash (most states). Send by registered post. What to write: Your name, address, description of information needed. No reason required. If you are a BPL card holder: RTI is FREE — attach a copy of your BPL card. First Appeal: If PIO does not reply in 30 days, or you are dissatisfied, file the First Appeal with the First Appellate Authority (Senior to PIO) within 30 days. Second Appeal: If First Appeal is unsatisfactory, file with the Central/State Information Commission within 90 days. Section 8(1) exemptions: National security, Cabinet papers, personal information with no public interest, trade secrets, information given in fiduciary capacity. Section 8(2): Public interest override — even exempt information can be disclosed if public interest outweighs harm. Penalty for PIO: ■250/day (up to ■25,000) for unjustified delay.'),
    KBEntry('go_public_services', 'governance_admin', 'Public services delivery rights and SPSA', 'States have Public Services (Rights of Citizens) Acts (e.g., UP RTPS Act, Maharashtra Public Services Guarantee Act) that specify timelines for delivery of government services. Examples: ration card (30 days), income certificate (15 days), caste certificate (10–30 days), voter ID (30 days). How it works: Every service has a designated officer and a specified time limit. If the service is not delivered on time: (1) File an appeal online/offline with the First Appellate Authority named in the Act. (2) Second appeal to the State Public Services Delivery Commission. Penalties: Officers who delay without reason can be fined (e.g., ■250/day up to ■5,000 in some states). Online services: Most states now provide services on a single portal (DigiSeva, Seva Sindhu, RTPS, Jan Mitra Kendra). CPGRAMS (Central Public Grievance Redress and Monitoring System): For central government services — file at pgportal.gov.in. Response within 30 days. State CM helplines: Most states have a 1076/1950 type helpline for public grievances.'),
    KBEntry('go_lokpal_lokayukta', 'governance_admin', 'Lokpal, Lokayukta and anti-corruption bodies', 'Lokpal of India: Handles complaints against central government employees (including Group A/B/C/D officers and PMs/Ministers) for corruption. File at lokpal.nic.in — complaint must be filed within 7 years of the alleged act. Lokpal cannot investigate judges or the judiciary. State Lokayukta: State-level anti-corruption ombudsman — handles complaints against state government employees. Each state has its own Lokayukta Act — powers vary. File at the state Lokayukta office or portal. CVC (Central Vigilance Commission): Handles vigilance matters for central government; file at cvc.gov.in. CBI (Central Bureau of Investigation): Investigates corruption in central government — cases referred by courts, government or CVC. FIR can be filed at any police station for corruption by central employees. Whistle Blower Protection: Whistle Blowers Protection Act 2014 protects disclosures to CVC — anonymous complaints allowed.'),
    KBEntry('go_caste_certificate', 'governance_admin', 'Caste, income and domicile certificates', 'SC/ST Certificate: Issued by the Tahsildar/SDM based on official Scheduled Caste/Tribe lists for your state. OBC Certificate: Issued by Tahsildar/SDM; must specify whether the applicant is creamy layer or non-creamy layer. Income Certificate: Issued by Tahsildar — based on self-declaration and verification. Usually required for scholarship, EWS certificate. EWS (Economically Weaker Section) Certificate: Annual family income below ■8 lakh + no owned property beyond specified limits — issued by Tahsildar. Domicile/Residence Certificate: Issued by Tahsildar — needed for state quota admissions. Online application: Most states now have online portals (check your state revenue department website). Documents typically required: Ration card, address proof, existing caste certificate of parent (for new issuance), Aadhaar. If certificate is delayed: File RTI for status; complain to SDM or District Collector. Fake caste certificates: Using a fake certificate for reservation benefits is a criminal offence — discovered through vigilance or RTI by rivals.'),
    KBEntry('go_gram_sabha', 'governance_admin', 'Panchayat, Gram Sabha and local body rights', 'Panchayati Raj Acts (each state): Gram Panchayat is the basic unit — elected for 5 years. Gram Sabha: All adult voters of a village — the supreme body for village-level decisions. Gram Sabha meetings: Must be held at least twice a year (many states mandate quarterly). All adult villagers can attend and vote. MGNREGA (Mahatma Gandhi National Rural Employment Guarantee Act): Every rural household has the right to 100 days of unskilled work per year. How to demand work: Apply in writing to the Gram Panchayat — work must commence within 15 days or unemployment allowance is payable. MGNREGA helpline: 1800-111-555. Ombudsman: each state has a MGNREGA Ombudsman. Social audit: Gram Sabha has the right to audit MGNREGA works and other public spending — attend social audits conducted by State Social Audit Units. Village development: Gram Sabha must approve development plans under PESA (Panchayats (Extension to Scheduled Areas) Act 1996) in scheduled tribal areas. Local body grievances: File at the Block Development Officer (BDO) or District Panchayat Officer.'),
    KBEntry('go_municipal_rights', 'governance_admin', 'Municipal corporation and urban local body rights', "Urban Local Bodies (ULBs) — Municipal Corporations, Municipalities and Town Panchayats — govern urban services. Services covered: Property tax, building permissions, water supply, solid waste collection, street lighting, birth/death certificates. Property tax dispute: If you believe your property tax assessment is wrong, file an objection before the assessment authority in your ULB. Building permission: Apply online to your municipal body. Unauthorized construction can be penalised — but prior to demolition, notice and hearing are mandatory. Birth/death certificate: Apply online or at the nearest municipal office. Registered within 21 days of birth/death — late registration requires Magistrate's order. Swachh Bharat Mission: Report garbage collection failure at the ULB complaint portal or SBM national app. AMRUT (Atal Mission for Rejuvenation and Urban Transformation): Infrastructure projects — your ULB must have a plan; attend ward committee meetings to participate. Complaint: Most ULBs have online portals and toll-free helplines. Escalate to the District Collector or state Urban Development Ministry if unresolved."),
    KBEntry('go_passport_visa', 'governance_admin', 'Passport, visa and foreign travel documents', "Passport application: Apply at passportindia.gov.in — select nearest Passport Seva Kendra (PSK). Documents required: Proof of identity (Aadhaar), date of birth (birth certificate/school certificate), address proof. Tatkaal scheme: Emergency passport in 1–3 days at a higher fee. Police verification: Mandatory for normal passports; Tatkaal requires self-declaration and post-issuance verification. Minor passport: Both parents' consent required (if divorced, single parent may need an affidavit). If passport is rejected: You will receive a rejection letter with reasons — submit additional documents or file a review request. OCI (Overseas Citizen of India) card: Non-resident Indians of Indian origin can apply at ociservices.gov.in — lifelong multiple-entry visa to India. Visa extensions: Apply at the Foreigners Regional Registration Office (FRRO) at frro.gov.in. Lost passport abroad: Report to the nearest Indian Embassy — emergency travel document issued. Passport helpline: 1800-258-1800."),
    KBEntry('go_aadhaar_rights', 'governance_admin', 'Aadhaar rights, biometric lock and corrections', 'Aadhaar Act 2016: Aadhaar is a unique 12-digit identity number issued by UIDAI. Voluntary use: Aadhaar cannot be made mandatory for any purpose other than where the Supreme Court permits (PDS, MGNREGA, IT filing). Biometric lock: Lock your biometrics at myaadhaar.uidai.gov.in or the UIDAI app — prevents biometric fraud. OTP-based Aadhaar: For sensitive transactions, use OTP instead of biometric. Virtual ID (VID): Generate a 16-digit VID that can be shared instead of the actual Aadhaar number — protects privacy. Update Aadhaar online: Name, address, DOB, gender can be updated online at myaadhaar.uidai.gov.in. Offline Aadhaar: Download a XML or secure PDF (masked Aadhaar) — share this instead of physical card. Report Aadhaar misuse: UIDAI helpline 1947 or helpdesk@uidai.gov.in. Aadhaar-bank linking: Mandatory for receiving government subsidies (DBT). If linking fails: visit your bank branch with Aadhaar.'),
    KBEntry('go_disaster_relief', 'governance_admin', 'Disaster relief rights and SDRF/NDRF', 'National Disaster Management Act 2005: State Disaster Response Fund (SDRF) and National Disaster Response Fund (NDRF): compensate people affected by floods, cyclones, earthquakes, landslides, droughts. SDRF compensation norms: Prescribed by government (cover: house damage, crop loss, loss of livelihood, death compensation). How to claim: Report to the village head/Gram Panchayat or urban ward officer immediately after the disaster. District Collector coordinates SDRF relief. PM Fasal Bima Yojana (Pradhan Mantri Crop Insurance): Farmers can claim insurance for crop loss due to natural calamity — register at pmfby.gov.in. NDRF (National Disaster Response Force): For rescue and relief in major disasters — call 112 or NDRF helpline 9711077372. Relief camp: State government must provide shelter, food, drinking water, medical care in relief camps. Grivance in relief: If relief is delayed or denied, approach the District Magistrate; file RTI for relief distribution records. PM SVANidhi: Micro-loan scheme for street vendors displaced by disasters (■10,000–50,000).'),
    KBEntry('go_right_to_protest', 'governance_admin', 'Right to protest and assembly', 'Article 19(1)(b): Right to assemble peaceably and without arms. CrPC Section 144 (BNSS Section 163): Executive Magistrate can impose prohibitory orders in areas to prevent disturbance — challenge this in the High Court if imposed without basis. Permission for procession/demonstration: Prior permission from local police/District Magistrate is required in most states for a public procession — apply in writing at least 72 hours in advance. Condition: Cannot carry arms; must follow route approved by police. Unlawful assembly: Five or more persons with an unlawful common object — police can disperse and arrest. Protest near Parliament/State Assembly: Prohibited under special rules during sessions. Tear gas, lathi charge: Permissible only if assembly is unlawful and officers have given due warning. Excessive force violates Article 21. If arrested at a protest: Exercise right to inform family (Article 22). Request a lawyer. Do not sign any statement without reading. Call NALSA 15100 for free legal aid.'),
    KBEntry('go_public_health', 'governance_admin', 'Public health rights and government healthcare', 'Public health is a state subject; central government sets policy through National Health Mission (NHM). Ayushman Bharat – PMJAY (Pradhan Mantri Jan Arogya Yojana): Up to ■5 lakh per family per year for hospitalisation at empanelled government and private hospitals. Check eligibility: pmjay.gov.in or call 14555. JSSK (Janani Shishu Suraksha Karyakram): Free maternal and newborn care at government facilities. No denial rule: Government hospitals cannot refuse emergency treatment — even without Aadhaar or insurance. Medicines: Free essential medicines in government hospitals under Pradhan Mantri Bhartiya Jan Aushadhi Pariyojana (PMBJP) — low-cost generic medicines. Mental health: Mental Healthcare Act 2017 — right to mental healthcare, community living, free legal aid for persons with mental illness. Mental Health helpline: iCall (9152987821), Vandrevala Foundation (1860-2662-345), NIMHANS (080-46110007). Complaint about government hospital: Write to Medical Superintendent → CMO → State Health Secretary.'),
    KBEntry('go_environment_ngt', 'governance_admin', 'NGT — how to file an environmental complaint', 'National Green Tribunal (NGT) Act 2010: A specialised court for environmental cases. Who can file: Any person, including public interest petitions. No need for a lawyer (but recommended). What can be filed: (1) Violation of environmental laws (EP Act, Water Act, Air Act, Forest Conservation Act). (2) Environmental damage caused by industry, construction, or government project. (3) Seeking compensation for damage caused by pollution. Filing: Online: ngtnational.gov.in → E-filing. Offline: Submit petition to Principal Bench (New Delhi) or regional bench. Filing fee: ■1,000 for an application. Powers: NGT can pass interim orders (stop polluting activity), award compensation, and impose penalties on polluters. Limitation: File within 3 years of the cause of action. CPCB/SPCB: Can also be approached directly before filing at NGT — they have compliance mechanisms.'),
    KBEntry('go_police_norms', 'governance_admin', 'Standards and laws governing police conduct', "Model Police Act 2006 (Prakash Singh judgment 2006): Supreme Court mandated police reforms — but implementation varies by state. Your rights during police interaction: (1) Police CANNOT demand your phone password or search your devices without a warrant (for most purposes). (2) Police CANNOT arrest on a civil dispute (e.g., non-payment of money) without a court order. (3) Police must give you reasons for arrest in writing within 24 hours. (4) If asked to come to the police station 'for questioning' — you are either a witness (cannot be detained) or suspect (must be shown the arrest memo). Custodial torture: Banned under Article 21, Convention Against Torture. Complaint to NHRC within 3 months. False FIR against you: Apply for anticipatory bail; file a complaint for malicious prosecution/perjury; challenge the FIR by filing a Writ Petition in the High Court. Police Complaints Authority (PCA): Established in some states as per Prakash Singh guidelines — independent body to hear complaints against police. Third-degree torture: File habeas corpus in High Court; also complain to NHRC and State Human Rights Commission."),
    KBEntry('go_social_welfare', 'governance_admin', 'Social welfare schemes and entitlements for vulnerable groups', "Antyodaya Anna Yojana: 35 kg of foodgrain/month at ■2–3/kg for the poorest families. NSAP (National Social Assistance Programme): Old-age pension, widow pension, disability pension for BPL families — apply at district social welfare office. Pradhan Mantri Awaas Yojana: Housing for EWS/LIG — apply at pmaymis.gov.in (urban) or pmayg.nic.in (rural). Ayushman Bharat: ■5 lakh health cover. Ujjwala Yojana: Free LPG connection for BPL women — apply at nearest LPG distributor. PM-KISAN: ■6,000/year to eligible farmers. National Family Benefit Scheme (NFBS): Lump sum ■20,000 to BPL families on death of primary breadwinner. Widow Remarriage Act 1856: Widow's remarriage is fully legal — they have right to maintenance and property from the first marriage. DBT (Direct Benefit Transfer): All central scheme benefits are sent directly to Aadhaar-linked bank accounts — ensure your Aadhaar and bank are linked to receive benefits. General Guidance 1 existing | 0 new | 1 total"),
    KBEntry('general', 'general', 'General guidance when situation is unclear', 'When facts are incomplete, the most useful next step is to document clearly: WHO was involved (names, positions, phone numbers). WHAT happened (specific events, words said, actions taken). WHEN (dates and times). WHERE (location, city, state). HOW MUCH (money lost, damage caused). EVIDENCE (screenshots, messages, receipts, photos, videos, witnesses). Immediate safety: If anyone is in danger right now, call 112 (Police Emergency) immediately. For women in danger: 181. For children: 1098 (CHILDLINE). For seniors: 14567. Free legal aid: NALSA helpline 15100 — can connect you to a free lawyer in your district. District Legal Services Authority (DLSA): Every district has one. They provide free legal consultation and representation to those who cannot afford a lawyer.'),
]

KB_DOMAIN_IDS: List[str] = sorted({e.domain for e in KNOWLEDGE_BASE} - {"general"})


@dataclass
class GoldCase:
    text: str
    correct_helplines: Set[str]
    correct_sections: Set[str]
    expected_domains: List[str]
    difficulty: str   


GOLD_DATASET: List[GoldCase] = [
    
    GoldCase(
        text="Someone called me pretending to be from my bank and asked for my OTP. I gave it and ₹45,000 was deducted from my account.",
        correct_helplines={"1930", "cybercrime.gov.in"},
        correct_sections={"Section 66C", "Section 66D"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="easy",
    ),
    GoldCase(
        text="My UPI account was hacked and ₹12,000 transferred without my permission.",
        correct_helplines={"1930"},
        correct_sections={"Section 66C"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="easy",
    ),
    GoldCase(
        text="Someone created a fake Instagram profile using my photos and is messaging my friends.",
        correct_helplines={"1930", "cybercrime.gov.in"},
        correct_sections={"Section 66D", "Section 66E"},
        expected_domains=["cyber_it"],
        difficulty="easy",
    ),
    GoldCase(
        text="I received a phishing link on WhatsApp and clicked it. Now my phone might be compromised.",
        correct_helplines={"1930"},
        correct_sections={"Section 66", "Section 43"},
        expected_domains=["cyber_it"],
        difficulty="medium",
    ),
    GoldCase(
        text="Someone is threatening to leak my private photos unless I pay them money.",
        correct_helplines={"1930", "1800-200-3323"},
        correct_sections={"Section 66E", "Section 67", "Section 308"},
        expected_domains=["cyber_it", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="Mere phone se kisi ne OTP le kar ₹30,000 UPI se transfer kar liya.",
        correct_helplines={"1930"},
        correct_sections={"Section 66C"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="easy",
    ),
    GoldCase(
        text="A loan app is accessing my contacts and threatening to message everyone if I don't pay.",
        correct_helplines={"1930", "sachet.rbi.org.in"},
        correct_sections={"Section 66", "Section 43"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="hard",
    ),
    GoldCase(
        text="My SIM was swapped without my knowledge. I stopped receiving OTPs and money was transferred.",
        correct_helplines={"1930"},
        correct_sections={"Section 66C", "Section 66D"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="hard",
    ),
    GoldCase(
        text="I got a message that I won a prize and must click a link to claim it. I clicked and entered my bank details.",
        correct_helplines={"1930"},
        correct_sections={"Section 66D", "Section 66C"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="easy",
    ),
    GoldCase(
        text="My email account was hacked and the hacker sent fraud messages to all my contacts asking for money.",
        correct_helplines={"1930", "cybercrime.gov.in"},
        correct_sections={"Section 66C", "Section 66D"},
        expected_domains=["cyber_it"],
        difficulty="medium",
    ),
    GoldCase(
        text="Someone installed spyware on my phone through a fake app. They are reading my messages.",
        correct_helplines={"1930"},
        correct_sections={"Section 66", "Section 43"},
        expected_domains=["cyber_it"],
        difficulty="medium",
    ),
    GoldCase(
        text="I was scammed by a fake job portal. I paid ₹15,000 as registration fee and they disappeared.",
        correct_helplines={"1930", "cybercrime.gov.in"},
        correct_sections={"Section 66D"},
        expected_domains=["cyber_it", "consumer"],
        difficulty="easy",
    ),
    GoldCase(
        text="My child's explicit images were taken and circulated on Telegram without consent. The person is unknown.",
        correct_helplines={"1930", "1098"},
        correct_sections={"Section 67B", "Section 66E"},
        expected_domains=["cyber_it", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="A former friend is running a defamation campaign against me on social media using fake screenshots.",
        correct_helplines={"1930"},
        correct_sections={"Section 66D", "Section 66"},
        expected_domains=["cyber_it", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="I received a QR code to collect a refund. I scanned it and ₹20,000 was deducted instead.",
        correct_helplines={"1930"},
        correct_sections={"Section 66C"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="easy",
    ),
    # ── BANKING / FINANCE ───────────────────────────────────────────────────
    GoldCase(
        text="₹8,000 was debited from my bank account by an unauthorized party. I reported it the next day.",
        correct_helplines={"cms.rbi.org.in"},
        correct_sections={"DBR.No.Leg.BC.78/09.07.005/2017-18"},
        expected_domains=["banking_finance"],
        difficulty="easy",
    ),
    GoldCase(
        text="Recovery agents from a finance company are calling me at midnight and threatening my family.",
        correct_helplines={"cms.rbi.org.in", "sachet.rbi.org.in"},
        correct_sections={"Section 296"},
        expected_domains=["banking_finance", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="My insurance claim was rejected without a proper explanation after a road accident.",
        correct_helplines={"155255", "1800-4254-732", "cioins.co.in"},
        correct_sections=set(),
        expected_domains=["banking_finance", "consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="The bank has not reversed a fraudulent transaction even after 45 days of my complaint.",
        correct_helplines={"cms.rbi.org.in"},
        correct_sections=set(),
        expected_domains=["banking_finance"],
        difficulty="medium",
    ),
    GoldCase(
        text="Credit card company is charging interest on a transaction I never made.",
        correct_helplines={"cms.rbi.org.in"},
        correct_sections=set(),
        expected_domains=["banking_finance", "consumer"],
        difficulty="easy",
    ),
    GoldCase(
        text="My bank account was frozen without notice and no reason was given. I cannot access my money.",
        correct_helplines={"cms.rbi.org.in"},
        correct_sections=set(),
        expected_domains=["banking_finance"],
        difficulty="medium",
    ),
    GoldCase(
        text="A private NBFC lender is charging 50% annual interest and seized my documents when I missed one EMI.",
        correct_helplines={"sachet.rbi.org.in", "cms.rbi.org.in"},
        correct_sections=set(),
        expected_domains=["banking_finance"],
        difficulty="hard",
    ),
    GoldCase(
        text="My health insurance company dishonoured a cashless claim at a network hospital saying it is excluded.",
        correct_helplines={"155255", "cioins.co.in"},
        correct_sections=set(),
        expected_domains=["banking_finance", "consumer"],
        difficulty="hard",
    ),
    GoldCase(
        text="I never took a loan but my CIBIL score dropped because of a loan entry from a bank I never approached.",
        correct_helplines={"cms.rbi.org.in"},
        correct_sections={"Section 66C"},
        expected_domains=["banking_finance", "cyber_it"],
        difficulty="hard",
    ),
    GoldCase(
        text="Bank charged me ₹5,000 in hidden fees not disclosed at account opening.",
        correct_helplines={"cms.rbi.org.in"},
        correct_sections=set(),
        expected_domains=["banking_finance", "consumer"],
        difficulty="easy",
    ),
    # ── LABOUR / EMPLOYMENT ─────────────────────────────────────────────────
    GoldCase(
        text="My employer has not paid my salary for the past 3 months and refuses to respond.",
        correct_helplines={"shramsuvidha.gov.in", "1800-118-005"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment"],
        difficulty="easy",
    ),
    GoldCase(
        text="I was fired without any notice period or severance pay after 4 years at the company.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Industrial Disputes Act 1947"},
        expected_domains=["labour_employment"],
        difficulty="easy",
    ),
    GoldCase(
        text="My PF contribution is not being deposited by my employer despite deducting it from my salary.",
        correct_helplines={"1800-118-005", "epfigms.gov.in"},
        correct_sections={"Code on Social Security 2020"},
        expected_domains=["labour_employment"],
        difficulty="medium",
    ),
    GoldCase(
        text="My senior manager at work is sexually harassing me and HR is ignoring my complaints.",
        correct_helplines={"181", "shebox.nic.in"},
        correct_sections={"POSH Act 2013"},
        expected_domains=["labour_employment", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="I am a gig worker for a delivery platform. I got injured while working and they are denying compensation.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Social Security 2020"},
        expected_domains=["labour_employment"],
        difficulty="hard",
    ),
    GoldCase(
        text="Salary nahi mili 3 mahine se. HR koi jawab nahi de raha.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment"],
        difficulty="easy",
    ),
    GoldCase(
        text="I am a domestic worker. My employer refuses to pay me for the last 2 months and threatens to call police.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Minimum Wages Act"},
        expected_domains=["labour_employment"],
        difficulty="medium",
    ),
    GoldCase(
        text="My company is making me work 14 hours a day with no overtime pay and no weekly off.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment"],
        difficulty="medium",
    ),
    GoldCase(
        text="I am on maternity leave but my employer terminated me and says I resigned.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Social Security 2020"},
        expected_domains=["labour_employment"],
        difficulty="hard",
    ),
    GoldCase(
        text="My employer is withholding my experience certificate and full-and-final settlement for 6 months after I resigned.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment"],
        difficulty="medium",
    ),
    GoldCase(
        text="I was a contract worker for 3 years. The principal employer says they are not responsible for my unpaid wages.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment"],
        difficulty="hard",
    ),
    GoldCase(
        text="My employer forced me to sign a blank resignation letter as a condition of joining. Now they are using it.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Industrial Disputes Act 1947"},
        expected_domains=["labour_employment"],
        difficulty="hard",
    ),
    # ── CRIMINAL LAW ────────────────────────────────────────────────────────
    GoldCase(
        text="My neighbor is threatening to kill me if I don't vacate my house. He has a weapon.",
        correct_helplines={"112"},
        correct_sections={"Section 296", "Section 109"},
        expected_domains=["criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="I was physically assaulted by my landlord when I asked for my deposit back.",
        correct_helplines={"112"},
        correct_sections={"Section 115", "Section 132"},
        expected_domains=["criminal_law", "civil_property"],
        difficulty="medium",
    ),
    GoldCase(
        text="A gang robbed me on the street and took my phone and wallet at knifepoint.",
        correct_helplines={"112"},
        correct_sections={"Section 303", "Section 308"},
        expected_domains=["criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="A man is following me home every day and sending threatening messages on Instagram.",
        correct_helplines={"1091", "112"},
        correct_sections={"Section 74", "Section 296"},
        expected_domains=["criminal_law", "cyber_it"],
        difficulty="medium",
    ),
    GoldCase(
        text="My business partner cheated me out of ₹5 lakh. He forged my signature on documents.",
        correct_helplines={"112"},
        correct_sections={"Section 316", "Section 317"},
        expected_domains=["criminal_law", "civil_property"],
        difficulty="hard",
    ),
    GoldCase(
        text="My brother was kidnapped and the kidnappers are demanding ₹10 lakh ransom.",
        correct_helplines={"112"},
        correct_sections={"Section 135", "Section 308"},
        expected_domains=["criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="My car was stolen from a parking lot last night.",
        correct_helplines={"112"},
        correct_sections={"Section 303"},
        expected_domains=["criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="I was offered a job and paid ₹50,000 as a security deposit. The company was fake.",
        correct_helplines={"112"},
        correct_sections={"Section 316"},
        expected_domains=["criminal_law", "consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="My colleague spread false rumours about me at work that damaged my professional reputation.",
        correct_helplines={"112"},
        correct_sections={"Section 296"},
        expected_domains=["criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="A moneylender who gave me an informal loan is now threatening me with acid attack if I don't repay immediately.",
        correct_helplines={"112"},
        correct_sections={"Section 296", "Section 124"},
        expected_domains=["criminal_law", "banking_finance"],
        difficulty="hard",
    ),
    # ── CONSUMER PROTECTION ─────────────────────────────────────────────────
    GoldCase(
        text="I ordered a laptop on Flipkart. It arrived damaged and they are refusing to refund.",
        correct_helplines={"1800-11-4000", "1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer"],
        difficulty="easy",
    ),
    GoldCase(
        text="A company charged me for a subscription I never signed up for. They won't refund.",
        correct_helplines={"1915"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer"],
        difficulty="easy",
    ),
    GoldCase(
        text="A car service centre did unnecessary repairs on my car and charged me double.",
        correct_helplines={"1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="The hospital performed a surgery on me without properly explaining the risks and getting my consent.",
        correct_helplines={"e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer", "governance_admin"],
        difficulty="hard",
    ),
    GoldCase(
        text="My insurance company is delaying settlement of my health claim for 60 days without reason.",
        correct_helplines={"155255", "cioins.co.in"},
        correct_sections={"IRDAI"},
        expected_domains=["consumer", "banking_finance"],
        difficulty="medium",
    ),
    GoldCase(
        text="I bought a new refrigerator and it stopped working within 2 weeks. The company says this is not covered by warranty.",
        correct_helplines={"1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer"],
        difficulty="easy",
    ),
    GoldCase(
        text="A travel agency took payment for a tour package but cancelled last-minute and refused to return money.",
        correct_helplines={"1915"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="A builder promised to deliver my flat in 2 years. It has been 5 years and the flat is still not complete.",
        correct_helplines={"e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer", "civil_property"],
        difficulty="hard",
    ),
    GoldCase(
        text="A gym charged me for a one-year membership upfront and shut down after 2 months.",
        correct_helplines={"1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="A coaching centre misrepresented their pass percentage. I paid ₹80,000 and the faculty was incompetent.",
        correct_helplines={"1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer", "education"],
        difficulty="hard",
    ),
    # ── CIVIL / PROPERTY ────────────────────────────────────────────────────
    GoldCase(
        text="My landlord is not returning my ₹50,000 security deposit even though I vacated 2 months ago.",
        correct_helplines=set(),
        correct_sections={"Model Tenancy Act 2021"},
        expected_domains=["civil_property"],
        difficulty="easy",
    ),
    GoldCase(
        text="My landlord locked my flat while I was out and threw away my belongings. He wants me out immediately.",
        correct_helplines={"112"},
        correct_sections={"BNS Section 329", "Model Tenancy Act 2021"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="Someone sold me a plot of land but it turns out they were not the real owner. I paid ₹8 lakh.",
        correct_helplines={"112"},
        correct_sections={"Section 316", "Transfer of Property Act 1882"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="My father died without a will. My brother is claiming sole ownership of the family house.",
        correct_helplines=set(),
        correct_sections={"Transfer of Property Act 1882"},
        expected_domains=["civil_property", "family_personal"],
        difficulty="hard",
    ),
    GoldCase(
        text="Mera zameen ka vivad chal raha hai. Padosi ne kabja kar liya hai.",
        correct_helplines={"112"},
        correct_sections={"BNS Section 329"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="My landlord is raising my rent by 40% mid-lease and threatening eviction if I don't agree.",
        correct_helplines=set(),
        correct_sections={"Model Tenancy Act 2021"},
        expected_domains=["civil_property"],
        difficulty="medium",
    ),
    GoldCase(
        text="My neighbour has encroached on 2 feet of my land and built a wall. I have the original sale deed.",
        correct_helplines={"112"},
        correct_sections={"Transfer of Property Act 1882"},
        expected_domains=["civil_property"],
        difficulty="medium",
    ),
    GoldCase(
        text="I gave a loan of ₹3 lakh to a friend. He has refused to return it and blocked me.",
        correct_helplines=set(),
        correct_sections={"Section 316"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="My builder registered the flat in my name but kept possession and is renting it to someone else.",
        correct_helplines={"112"},
        correct_sections={"Transfer of Property Act 1882", "Section 316"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="A power of attorney holder sold my mother's property without her knowledge while she was in hospital.",
        correct_helplines={"112"},
        correct_sections={"Transfer of Property Act 1882", "Section 316"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="hard",
    ),
    # ── FAMILY / PERSONAL ───────────────────────────────────────────────────
    GoldCase(
        text="My husband beats me regularly. I have two children. I am afraid to leave.",
        correct_helplines={"181", "112", "15100"},
        correct_sections={"PWDVA 2005", "Section 498A"},
        expected_domains=["family_personal", "criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="My in-laws are demanding more dowry and have been physically abusing me.",
        correct_helplines={"181", "112"},
        correct_sections={"Dowry Prohibition Act 1961", "Section 498A"},
        expected_domains=["family_personal", "criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="My husband left me and our 3-year-old child. He is not paying any maintenance.",
        correct_helplines={"181", "15100"},
        correct_sections={"BNSS Section 144"},
        expected_domains=["family_personal"],
        difficulty="medium",
    ),
    GoldCase(
        text="My elderly mother is being neglected by my siblings who took her property.",
        correct_helplines={"14567"},
        correct_sections={"Maintenance and Welfare of Parents and Senior Citizens Act 2007"},
        expected_domains=["family_personal"],
        difficulty="medium",
    ),
    GoldCase(
        text="I want to divorce my husband. He is emotionally abusive but not physically violent.",
        correct_helplines={"181", "15100"},
        correct_sections={"PWDVA 2005"},
        expected_domains=["family_personal"],
        difficulty="medium",
    ),
    GoldCase(
        text="My father-in-law is threatening to kill me unless I sign over my share of the property.",
        correct_helplines={"112", "181"},
        correct_sections={"Section 296", "PWDVA 2005"},
        expected_domains=["family_personal", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="I am a minor and my parents are forcing me into marriage next month.",
        correct_helplines={"1098", "112"},
        correct_sections={"Section 498A"},
        expected_domains=["family_personal", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="My ex-partner is harassing me after separation and threatening to release private photos.",
        correct_helplines={"1930", "181"},
        correct_sections={"Section 66E", "Section 67", "PWDVA 2005"},
        expected_domains=["family_personal", "cyber_it"],
        difficulty="hard",
    ),
    GoldCase(
        text="My adult children are not supporting me financially. I am 70 years old and have no income.",
        correct_helplines={"14567"},
        correct_sections={"Maintenance and Welfare of Parents and Senior Citizens Act 2007"},
        expected_domains=["family_personal"],
        difficulty="easy",
    ),
    GoldCase(
        text="My wife has taken our child abroad on a tourist visa and refused to return. She ignores court orders.",
        correct_helplines={"181"},
        correct_sections={"BNSS Section 144"},
        expected_domains=["family_personal", "criminal_law"],
        difficulty="hard",
    ),
    # ── FUNDAMENTAL RIGHTS ──────────────────────────────────────────────────
    GoldCase(
        text="I was detained by police for 36 hours without being presented to a magistrate or told the reason.",
        correct_helplines={"15100", "14433"},
        correct_sections={"Article 22", "BNSS 2023"},
        expected_domains=["fundamental_rights", "governance_admin"],
        difficulty="medium",
    ),
    GoldCase(
        text="Upper caste people in my village are preventing SC community members from using the common well.",
        correct_helplines={"14566"},
        correct_sections={"Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989"},
        expected_domains=["fundamental_rights"],
        difficulty="medium",
    ),
    GoldCase(
        text="My employer refused to hire me because of my religion. They said so openly.",
        correct_helplines={"14433"},
        correct_sections={"Article 15", "Article 16"},
        expected_domains=["fundamental_rights", "labour_employment"],
        difficulty="hard",
    ),
    GoldCase(
        text="A company leaked my Aadhaar and biometric data to third parties without my consent.",
        correct_helplines={"1947"},
        correct_sections={"Article 21", "K.S. Puttaswamy v. Union of India 2017"},
        expected_domains=["fundamental_rights", "cyber_it"],
        difficulty="hard",
    ),
    GoldCase(
        text="Police arrested me without a warrant and refused to let me call my lawyer.",
        correct_helplines={"15100"},
        correct_sections={"Article 22", "BNSS 2023"},
        expected_domains=["fundamental_rights", "governance_admin"],
        difficulty="medium",
    ),
    GoldCase(
        text="Our village has been denied access to a government school for 3 years. Children walk 10 km.",
        correct_helplines=set(),
        correct_sections={"RTE Act 2009", "Article 21"},
        expected_domains=["fundamental_rights", "education"],
        difficulty="hard",
    ),
    GoldCase(
        text="A government colony is blocking entry of Dalit residents. Security guards are enforcing it.",
        correct_helplines={"14566", "14433"},
        correct_sections={"Article 14", "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989"},
        expected_domains=["fundamental_rights"],
        difficulty="hard",
    ),
    GoldCase(
        text="Municipal authorities demolished my shop without notice, saying it is illegal. I have all papers.",
        correct_helplines={"14433"},
        correct_sections={"Article 21", "Article 14"},
        expected_domains=["fundamental_rights", "governance_admin"],
        difficulty="hard",
    ),
    GoldCase(
        text="My organisation was denied permission to protest peacefully by police without any legal reason.",
        correct_helplines={"15100"},
        correct_sections={"Article 19"},
        expected_domains=["fundamental_rights"],
        difficulty="medium",
    ),
    GoldCase(
        text="My son with disability was denied admission to a government school citing inability to accommodate.",
        correct_helplines={"14433"},
        correct_sections={"Article 21", "RTE Act 2009"},
        expected_domains=["fundamental_rights", "education"],
        difficulty="medium",
    ),
    # ── GOVERNANCE / ADMIN ──────────────────────────────────────────────────
    GoldCase(
        text="Police are refusing to register my FIR even though I have been threatened.",
        correct_helplines={"112"},
        correct_sections={"S.154 BNSS", "S.175 BNSS"},
        expected_domains=["governance_admin", "criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="A government official is demanding a bribe to process my passport application.",
        correct_helplines={"cvc.gov.in", "lokpal.nic.in"},
        correct_sections={"Prevention of Corruption Act 1988"},
        expected_domains=["governance_admin"],
        difficulty="easy",
    ),
    GoldCase(
        text="I filed an RTI 45 days ago but the department has not replied.",
        correct_helplines={"rtionline.gov.in"},
        correct_sections={"RTI Act 2005"},
        expected_domains=["governance_admin"],
        difficulty="easy",
    ),
    GoldCase(
        text="My Aadhaar update is pending for 6 months. The office says come back next month every time.",
        correct_helplines={"1947", "uidai.gov.in"},
        correct_sections=set(),
        expected_domains=["governance_admin"],
        difficulty="easy",
    ),
    GoldCase(
        text="A police officer assaulted me in custody. I have bruises and a witness.",
        correct_helplines={"nhrc.nic.in", "14433"},
        correct_sections={"Article 21", "Article 22"},
        expected_domains=["governance_admin", "fundamental_rights"],
        difficulty="hard",
    ),
    GoldCase(
        text="The Municipal Corporation is not collecting garbage from our colony for 3 months.",
        correct_helplines={"rtionline.gov.in"},
        correct_sections={"RTI Act 2005"},
        expected_domains=["governance_admin"],
        difficulty="easy",
    ),
    GoldCase(
        text="My BPL ration card was cancelled without any reason and I cannot get food from PDS.",
        correct_helplines={"rtionline.gov.in"},
        correct_sections={"RTI Act 2005"},
        expected_domains=["governance_admin"],
        difficulty="medium",
    ),
    GoldCase(
        text="Sarkari daftar mein aadhi certificate nahi de raha bina paanch hazaar rupay ke.",
        correct_helplines={"cvc.gov.in"},
        correct_sections={"Prevention of Corruption Act 1988"},
        expected_domains=["governance_admin"],
        difficulty="easy",
    ),
    GoldCase(
        text="Police filed a case against me that I know was filed by my political opponent who is a local leader.",
        correct_helplines={"nhrc.nic.in", "14433"},
        correct_sections={"Article 21"},
        expected_domains=["governance_admin", "fundamental_rights"],
        difficulty="hard",
    ),
    GoldCase(
        text="My voter ID was deleted from the rolls before the election without any notice to me.",
        correct_helplines={"1950"},
        correct_sections=set(),
        expected_domains=["governance_admin", "fundamental_rights"],
        difficulty="medium",
    ),
    GoldCase(
        text="The government acquired my farmland for a highway but has not paid the compensation after 2 years.",
        correct_helplines={"rtionline.gov.in"},
        correct_sections={"Article 21"},
        expected_domains=["governance_admin", "civil_property"],
        difficulty="hard",
    ),
    # ── EDUCATION ───────────────────────────────────────────────────────────
    GoldCase(
        text="My college is withholding my degree certificate because of an outstanding fee dispute.",
        correct_helplines={"e-daakhil.nic.in"},
        correct_sections={"RTE Act 2009"},
        expected_domains=["education", "consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="Senior students in my hostel are ragging me physically and mentally.",
        correct_helplines={"1800-180-5522"},
        correct_sections={"UGC Regulations on Ragging 2009"},
        expected_domains=["education", "criminal_law"],
        difficulty="easy",
    ),
    GoldCase(
        text="My private school is charging fees way above the regulated limit.",
        correct_helplines={"e-daakhil.nic.in"},
        correct_sections={"RTE Act 2009"},
        expected_domains=["education", "consumer"],
        difficulty="medium",
    ),
    GoldCase(
        text="My central government scholarship has not been credited for 8 months.",
        correct_helplines={"scholarships.gov.in"},
        correct_sections=set(),
        expected_domains=["education", "governance_admin"],
        difficulty="medium",
    ),
    GoldCase(
        text="A teacher in my school is physically punishing children in class, including slapping.",
        correct_helplines={"1098", "112"},
        correct_sections={"RTE Act 2009"},
        expected_domains=["education", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="My college is conducting a screening test at Class 1 admission which I believe is illegal.",
        correct_helplines=set(),
        correct_sections={"RTE Act 2009"},
        expected_domains=["education"],
        difficulty="hard",
    ),
    GoldCase(
        text="My university cancelled my exam result with no explanation 3 months after declaration.",
        correct_helplines={"e-daakhil.nic.in"},
        correct_sections={"UGC Regulations on Ragging 2009"},
        expected_domains=["education", "governance_admin"],
        difficulty="hard",
    ),
    # ── MULTI-DOMAIN / HARD ─────────────────────────────────────────────────
    GoldCase(
        text="My employer fired me after I refused to pay him a bribe. Now he filed a false FIR against me.",
        correct_helplines={"shramsuvidha.gov.in", "cvc.gov.in", "112"},
        correct_sections={"Industrial Disputes Act 1947", "Prevention of Corruption Act 1988"},
        expected_domains=["labour_employment", "governance_admin", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="I transferred ₹2 lakh for a property deal. The seller disappeared. The property was fake.",
        correct_helplines={"1930", "112"},
        correct_sections={"Section 316", "Section 66C"},
        expected_domains=["civil_property", "criminal_law", "banking_finance"],
        difficulty="hard",
    ),
    GoldCase(
        text="My husband took my children abroad and refuses to let me see them. He is threatening me.",
        correct_helplines={"181", "112"},
        correct_sections={"PWDVA 2005", "BNSS Section 144"},
        expected_domains=["family_personal", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="Factory workers in our area are being paid below minimum wage and threatened with firing if they complain.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020", "Industrial Disputes Act 1947"},
        expected_domains=["labour_employment", "fundamental_rights"],
        difficulty="hard",
    ),
    GoldCase(
        text="Online pharmacy sent me expired medicines. I got sick. They are ignoring my complaints.",
        correct_helplines={"1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer", "criminal_law"],
        difficulty="medium",
    ),
    GoldCase(
        text="A private hospital refused emergency treatment to my mother because we couldn't pay upfront. She died.",
        correct_helplines={"14433", "nhrc.nic.in"},
        correct_sections={"Article 21"},
        expected_domains=["fundamental_rights", "consumer", "governance_admin"],
        difficulty="hard",
    ),
    GoldCase(
        text="I lost ₹10 lakh in a crypto investment scheme that turned out to be a Ponzi. My bank also froze my account saying my transactions look suspicious.",
        correct_helplines={"1930", "cms.rbi.org.in"},
        correct_sections={"Section 66D", "Section 66C"},
        expected_domains=["cyber_it", "banking_finance", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="My landlord is from upper caste and refuses to rent to me after learning my caste. He returned the advance.",
        correct_helplines={"14566"},
        correct_sections={"Article 15", "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989"},
        expected_domains=["fundamental_rights", "civil_property"],
        difficulty="hard",
    ),
    GoldCase(
        text="My son got injured in a road accident caused by a government bus. The driver fled. The hospital wants money upfront.",
        correct_helplines={"112", "14433"},
        correct_sections={"Article 21"},
        expected_domains=["fundamental_rights", "governance_admin", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="A social media influencer used my creative work without credit and is monetising it. The platform refuses to take action.",
        correct_helplines={"1930"},
        correct_sections={"Section 66"},
        expected_domains=["cyber_it", "consumer"],
        difficulty="hard",
    ),
    GoldCase(
        text="My company deducted TDS from my salary but never deposited it with the government. I cannot file my returns.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment", "governance_admin"],
        difficulty="hard",
    ),
    GoldCase(
        text="I am an inter-caste married couple. My in-laws filed a fake kidnapping case against me. My wife is detained.",
        correct_helplines={"112", "15100", "14566"},
        correct_sections={"Article 21", "BNSS 2023"},
        expected_domains=["fundamental_rights", "criminal_law", "family_personal"],
        difficulty="hard",
    ),
    GoldCase(
        text="My employer is deducting salary for every absence even though the policy says only after 3 absences are unpaid.",
        correct_helplines={"shramsuvidha.gov.in"},
        correct_sections={"Code on Wages 2020"},
        expected_domains=["labour_employment"],
        difficulty="medium",
    ),
    GoldCase(
        text="I reported corruption in my office through internal channels. Now I am being transferred to a remote location as punishment.",
        correct_helplines={"cvc.gov.in", "lokpal.nic.in"},
        correct_sections={"Prevention of Corruption Act 1988"},
        expected_domains=["governance_admin", "labour_employment"],
        difficulty="hard",
    ),
    GoldCase(
        text="My building's developer has collected maintenance fees for 5 years but the apartment association has no accounts.",
        correct_helplines={"1915", "e-daakhil.nic.in"},
        correct_sections={"Consumer Protection Act 2019"},
        expected_domains=["consumer", "civil_property"],
        difficulty="hard",
    ),
    GoldCase(
        text="A person I met online convinced me to invest in a trading app. I deposited ₹1.5 lakh and now cannot withdraw.",
        correct_helplines={"1930", "cms.rbi.org.in"},
        correct_sections={"Section 66D"},
        expected_domains=["cyber_it", "banking_finance"],
        difficulty="medium",
    ),
    GoldCase(
        text="I complained to my college's anti-ragging committee and was then threatened by senior students saying they will get me expelled.",
        correct_helplines={"1800-180-5522", "112"},
        correct_sections={"UGC Regulations on Ragging 2009", "Section 296"},
        expected_domains=["education", "criminal_law"],
        difficulty="hard",
    ),
    GoldCase(
        text="My father-in-law verbally abuses my husband to prevent him from talking to me. We live in his house.",
        correct_helplines={"181", "15100"},
        correct_sections={"PWDVA 2005"},
        expected_domains=["family_personal"],
        difficulty="medium",
    ),
    GoldCase(
        text="A neighbour is running an illegal factory next to my house producing toxic fumes. The municipality is ignoring complaints.",
        correct_helplines={"rtionline.gov.in"},
        correct_sections={"Article 21"},
        expected_domains=["fundamental_rights", "governance_admin"],
        difficulty="medium",
    ),
    GoldCase(
        text="I received a legal notice claiming I owe a debt to a company I never dealt with. They are threatening court action.",
        correct_helplines=set(),
        correct_sections={"Section 316"},
        expected_domains=["civil_property", "criminal_law"],
        difficulty="medium",
    ),
]

_HELPLINE_PATTERN = re.compile(
    # 3-digit emergency shortcodes — were missed entirely by the old 4-5 digit rule
    r"\b112\b|\b181\b"
    r"|\b1800[-\s]?\d{2,4}[-\s]?\d{3,4}\b"
    # other short/long helpline codes, e.g. 1098, 1930, 14433, 15100, 155255
    r"|\b1[0-9]{3,5}\b"
    r"|(?:[\w.-]+\.(?:gov|nic|org)\.in(?:/[\w./\-]*)?)"
    r"|cms\.rbi\.org\.in"
    r"|cioins\.co\.in",
    re.IGNORECASE,
)

_SECTION_PATTERN = re.compile(
    r"\bSection\s+\d+[A-Z]?\b"
    r"|\bArticle\s+\d+\b"
    r"|\bBNSS?\s+Section\s+\d+[A-Z]?\b"
    r"|\b(?:POSH|RTE|PWDVA|IPC|CrPC|BNSS|BNS|BSA)\b"
    r"|\b(?:IT\s+Act|Consumer\s+Protection\s+Act|Industrial\s+Disputes\s+Act"
    r"|Dowry\s+Prohibition\s+Act|Code\s+on\s+(?:Wages|Social\s+Security)"
    r"|Transfer\s+of\s+Property\s+Act|Maintenance\s+and\s+Welfare\s+of\s+Parents)"
    r"\s*(?:\d{4})?",
    re.IGNORECASE,
)


def _extract_helplines(text: str) -> Set[str]:
    """Extract helpline numbers / URLs from model output text."""
    return {m.group(0).strip().lower() for m in _HELPLINE_PATTERN.finditer(text)}


def _extract_sections(text: str) -> Set[str]:
    """Extract law section / article / act references from model output text."""
    return {m.group(0).strip() for m in _SECTION_PATTERN.finditer(text)}


def _compute_prf1(predicted: Set[str], gold: Set[str]) -> Dict[str, Optional[float]]:
    """Compute precision, recall, F1 and extra (potentially hallucinated) items."""
    if not predicted and not gold:
        return {"precision": None, "recall": None, "f1": None, "extra": []}
    if not gold:
        return {"precision": 0.0, "recall": None, "f1": None, "extra": sorted(predicted)}
    if not predicted:
        return {"precision": None, "recall": 0.0, "f1": 0.0, "extra": []}

    # Normalise: lower-case for helplines (URLs), keep original case for sections
    pred_lower = {p.lower() for p in predicted}
    gold_lower = {g.lower() for g in gold}

    tp = len(pred_lower & gold_lower)
    precision = tp / len(pred_lower) if pred_lower else 0.0
    recall = tp / len(gold_lower) if gold_lower else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    extra = sorted(pred_lower - gold_lower)
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "extra": extra}


def hallucination_scores(
    response_text: str,
    gold: GoldCase,
) -> Dict[str, Any]:
    
    pred_helplines = _extract_helplines(response_text)
    pred_sections = _extract_sections(response_text)

    hl = _compute_prf1(pred_helplines, gold.correct_helplines)
    hl["predicted"] = sorted(pred_helplines)
    hl["gold"] = sorted(gold.correct_helplines)

    sec = _compute_prf1(pred_sections, gold.correct_sections)
    sec["predicted"] = sorted(pred_sections)
    sec["gold"] = sorted(gold.correct_sections)

    return {"helplines": hl, "sections": sec}


EVAL_DIMS = ("grounding", "actionability", "hallucination", "relevance")


def cohens_kappa(ratings_a: List[int], ratings_b: List[int]) -> Optional[float]:
    
    if len(ratings_a) != len(ratings_b) or not ratings_a:
        return None

    n = len(ratings_a)
    observed_agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    # Marginal distributions
    from collections import Counter
    counts_a = Counter(ratings_a)
    counts_b = Counter(ratings_b)
    all_cats = set(counts_a) | set(counts_b)

    # Expected agreement by chance
    expected_agree = sum(
        (counts_a.get(cat, 0) / n) * (counts_b.get(cat, 0) / n)
        for cat in all_cats
    )

    if expected_agree >= 1.0:
        return None  

    return round((observed_agree - expected_agree) / (1.0 - expected_agree), 4)


def kappa_across_cases(
    judge1_scores: List[Dict[str, int]],
    judge2_scores: List[Dict[str, int]],
) -> Dict[str, Optional[float]]:
    
    kappas: Dict[str, Optional[float]] = {}
    for dim in EVAL_DIMS:
        a = [s.get(dim, 0) for s in judge1_scores if dim in s]
        b = [s.get(dim, 0) for s in judge2_scores if dim in s]
        min_len = min(len(a), len(b))
        kappas[dim] = cohens_kappa(a[:min_len], b[:min_len])
    return kappas



# ---------------------------------------------------------------------------
# KNOWLEDGE BASE 
# ---------------------------------------------------------------------------

KB_DOMAIN_IDS = sorted({e.domain for e in KNOWLEDGE_BASE} - {"general"})

print(f"[rag_engine] Knowledge base loaded: {len(KNOWLEDGE_BASE)} entries "
      f"across {len(KB_DOMAIN_IDS)} domains.")

_kb_docs = [e.content for e in KNOWLEDGE_BASE]

_vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                            sublinear_tf=True, max_features=60_000)
_vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                            sublinear_tf=True, max_features=20_000)

_X_char = _vec_char.fit_transform(_kb_docs)
_X_word = _vec_word.fit_transform(_kb_docs)

# ── ChromaDB vector store ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./nyaya_chroma_db")
CHROMA_COLLECTION  = "nyaya_kb"
EMBED_MODEL_NAME   = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
HYBRID_ALPHA       = float(os.environ.get("HYBRID_ALPHA", "0.65"))  

_chroma_client: Optional[Any] = None
_chroma_collection: Optional[Any] = None
_embed_model: Optional[Any] = None


def _get_chroma() -> Optional[Any]:
    """Lazy-init ChromaDB client + collection.  Returns None if unavailable."""
    global _chroma_client, _chroma_collection, _embed_model
    if not _CHROMA_AVAILABLE:
        return None
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        existing = [c.name for c in _chroma_client.list_collections()]
        if CHROMA_COLLECTION in existing:
            _chroma_collection = _chroma_client.get_collection(CHROMA_COLLECTION)
            print(f"[rag_engine] Loaded ChromaDB collection '{CHROMA_COLLECTION}' "
                  f"({_chroma_collection.count()} docs).")
        else:
            _chroma_collection = _build_chroma_collection()
        return _chroma_collection
    except Exception as exc:
        print(f"[rag_engine] ChromaDB init failed: {exc} — using TF-IDF only.")
        return None


def _build_chroma_collection() -> Any:
    
    print(f"[rag_engine] Building ChromaDB collection '{CHROMA_COLLECTION}'…")
    col = _chroma_client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    texts = [e.content for e in KNOWLEDGE_BASE]
    embeddings = _embed_model.encode(texts, show_progress_bar=False).tolist()
    col.add(
        ids=[e.id for e in KNOWLEDGE_BASE],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"id": e.id, "domain": e.domain, "title": e.title}
            for e in KNOWLEDGE_BASE
        ],
    )
    print(f"[rag_engine] Indexed {len(KNOWLEDGE_BASE)} KB entries into ChromaDB.")
    return col


def _semantic_scores(query: str, idx: List[int]) -> np.ndarray:
    
    col = _get_chroma()
    if col is None or _embed_model is None:
        return np.zeros(len(idx))

    # Query embedding
    q_emb = _embed_model.encode([query], show_progress_bar=False).tolist()

    # IDs of the subset we're allowed to retrieve from
    allowed_ids = [KNOWLEDGE_BASE[i].id for i in idx]

    try:
        results = col.query(
            query_embeddings=q_emb,
            n_results=len(allowed_ids),
            where={"id": {"$in": allowed_ids}} if len(allowed_ids) < len(KNOWLEDGE_BASE) else None,
            include=["distances"],
        )
        # ChromaDB returns distance (lower = closer for cosine)
        dist_map: Dict[str, float] = {}
        if results["ids"] and results["ids"][0]:
            for rid, dist in zip(results["ids"][0], results["distances"][0]):
                dist_map[rid] = 1.0 - float(dist)  # cosine similarity

        return np.array([dist_map.get(KNOWLEDGE_BASE[i].id, 0.0) for i in idx])
    except Exception as exc:
        print(f"[rag_engine] ChromaDB query error: {exc}")
        return np.zeros(len(idx))


def retrieve(query: str, domain_ids: Optional[List[str]] = None, top_k: int = 5) -> List[Tuple[KBEntry, float]]:
    
    if domain_ids:
        wanted = set(domain_ids) | {"general"}
        idx = [i for i, e in enumerate(KNOWLEDGE_BASE) if e.domain in wanted]
    else:
        idx = list(range(len(KNOWLEDGE_BASE)))

    if not idx:
        idx = list(range(len(KNOWLEDGE_BASE)))

    # ── Lexical (TF-IDF) ──────────────────────────────────────────────────
    q_char = _vec_char.transform([query])
    q_word = _vec_word.transform([query])
    sims_char = cosine_similarity(q_char, _X_char[idx])[0]
    sims_word = cosine_similarity(q_word, _X_word[idx])[0]
    lexical = sims_char * 0.6 + sims_word * 0.4

    # ── Semantic (ChromaDB) ───────────────────────────────────────────────
    col = _get_chroma()
    if col is not None:
        semantic = _semantic_scores(query, idx)
        sims = HYBRID_ALPHA * semantic + (1 - HYBRID_ALPHA) * lexical
    else:
        sims = lexical

    order = np.argsort(sims)[::-1][:top_k]
    return [(KNOWLEDGE_BASE[idx[i]], float(sims[i])) for i in order]


def _entry_to_document(entry: KBEntry, score: float) -> Document:
    return Document(
        page_content=entry.content,
        metadata={"id": entry.id, "domain": entry.domain, "title": entry.title, "score": score},
    )


class KnowledgeBaseRetriever(BaseRetriever):
   
    domain_ids: Optional[List[str]] = None
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = retrieve(query, domain_ids=self.domain_ids, top_k=self.top_k)
        return [_entry_to_document(entry, score) for entry, score in results]

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not _GEMINI_API_KEY:
    print("[rag_engine] GEMINI_API_KEY is not set — AI generation will be "
          "unavailable until it's configured.")

# Primary model for all generation tasks
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_configured = False
_gemini: Optional[Any] = None


def _get_gemini_model() -> Optional[Any]:
    
    global _gemini_configured, _gemini
    if _gemini is not None:
        return _gemini
    if not _GEMINI_API_KEY:
        return None
    try:
        if not _gemini_configured:
            _gemini = genai.Client(api_key=_GEMINI_API_KEY)
            _gemini_configured = True
        return _gemini
    except Exception as exc:
        print(f"[rag_engine] Gemini init failed: {exc}")
        return None
    
def _parse_retry_delay_seconds(exc: Exception) -> Optional[float]:
    
    msg = str(exc)
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _is_rate_limit_or_overload_error(exc: Exception) -> bool:
    
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "503" in msg
        or "UNAVAILABLE" in msg
        or "overloaded" in msg.lower()
    )



GEMINI_MAX_RETRIES: int = int(os.environ.get("GEMINI_MAX_RETRIES", "8"))
GEMINI_BACKOFF_BASE_S: float = float(os.environ.get("GEMINI_BACKOFF_BASE_S", "2.0"))
GEMINI_BACKOFF_MAX_S: float = float(os.environ.get("GEMINI_BACKOFF_MAX_S", "60.0"))


def generate_explanation(prompt: str, temperature: float = 0.7) -> Optional[str]:
   
    client = _get_gemini_model()
    if client is None:
        return None

    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": temperature},
            )
            return response.text
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_or_overload_error(exc):
                print(f"[rag_engine] generate_explanation failed (non-retryable): {exc}")
                return None
            if attempt == GEMINI_MAX_RETRIES:
                break
            delay = _parse_retry_delay_seconds(exc)
            if delay is None:
                delay = min(GEMINI_BACKOFF_BASE_S * (2 ** attempt), GEMINI_BACKOFF_MAX_S)
            else:
                delay = min(delay + 0.5, GEMINI_BACKOFF_MAX_S)  # small safety margin
            print(
                f"[rag_engine] generate_explanation rate-limited/overloaded "
                f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES}); retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    print(f"[rag_engine] generate_explanation failed after {GEMINI_MAX_RETRIES} retries: {last_exc}")
    return None
    


# ── Vector store / embeddings ─────────────────────────────────────────────────

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_REFINEMENT_ROUNDS: int = 2
MAX_TOTAL_PASSAGES: int = 12
TOP_K_PASSAGES: int = 5

VALID_AGENT_ACTIONS = {"answer", "clarify", "search"}

UNAVAILABLE_MESSAGE = (
    "I'm unable to generate a full AI response right now. "
    "For free legal aid, call NALSA: **15100**. "
    "For emergencies, call **112**."
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CORE LLM 
# ─────────────────────────────────────────────────────────────────────────────

def get_rag_explanation(
    user_text: str,
    primary_category: str,
    risk_level: str,
    domain_ids: Optional[List[str]] = None,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:

    passages = retrieve(
        query=user_text,
        domain_ids=domain_ids,
        top_k=5,
    )

    sources = [
        {
            "title": entry.title,
            "snippet": entry.content[:300] + "..."
            if len(entry.content) > 300
            else entry.content,
        }
        for entry, _ in passages
    ]

    context = "\n\n".join(
        [
            f"[{entry.title}]\n{entry.content}"
            for entry, _ in passages
        ]
    )

    prompt = f"""
You are Nyaya AI, an educational civic-awareness assistant.

User issue:
{user_text}

Detected category: {primary_category}
Risk level: {risk_level}

Relevant knowledge base:

{context}

Instructions:
- Answer only using the provided knowledge.
- Do not invent laws, helplines, or procedures.
- Clearly state uncertainty if information is missing.
- Keep the response educational, not legal advice.
"""

    explanation = generate_explanation(prompt)

    if explanation is None:
        return {
            "ai_available": False,
            "ai_explanation": UNAVAILABLE_MESSAGE,
            "ai_sources": sources,
            "self_check": {
                "ok": None,
                "issues": [],
                "revised": False,
            },
        }

    return {
        "ai_available": True,
        "ai_explanation": explanation,
        "ai_sources": sources,
        "self_check": {
            "ok": None,
            "issues": [],
            "revised": False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KNOWLEDGE BASE  
# ─────────────────────────────────────────────────────────────────────────────

class Passage:
    """Lightweight document wrapper used by the orchestration layer below."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata   # has: id, title, domain, score


class KnowledgeBase:
    

    def search(self, query: str, domain: Optional[str] = None, top_k: int = TOP_K_PASSAGES) -> List[Passage]:
        domain_ids = None if not domain or domain == "general" else [domain]
        results = retrieve(query, domain_ids=domain_ids, top_k=top_k)
        return [
            Passage(
                page_content=entry.content,
                metadata={"id": entry.id, "domain": entry.domain, "title": entry.title, "score": score},
            )
            for entry, score in results
        ]


# Global KB instance — backed by the real KNOWLEDGE_BASE, no loading step needed.
_kb = KnowledgeBase()


def _passages_to_sources(passages: List[Passage]) -> List[Dict[str, str]]:
    seen, sources = set(), []
    for p in passages:
        title = p.metadata.get("title", "")
        if title in seen:
            continue
        seen.add(title)
        domain = p.metadata.get("domain", "")
        snippet_text = p.page_content[:200] + ("…" if len(p.page_content) > 200 else "")
        sources.append({"title": title, "domain": domain, "snippet": snippet_text})
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — JSON UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> Any:
    """Extract the first JSON object or array from a string."""
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in text.")
    return json.loads(match.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HISTORY FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _format_history(conversation: List[Dict[str, str]]) -> str:
    lines = []
    for msg in conversation[-6:]:          # last 3 turns keeps context short
        role = msg.get("role", "user").capitalize()
        lines.append(f"{role}: {msg.get('content', '').strip()}")
    return "\n".join(lines) if lines else "(no previous turns)"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PLANNER
# ─────────────────────────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are the planning layer of Nyaya AI, an Indian legal-awareness assistant.

Conversation so far:
{history}

New user message:
{user_message}

Task: Decide what to search for in our knowledge base.
Valid domains: {domain_list}

Return ONLY valid JSON — an object with two keys:
1. "queries": list of objects, each with "query" (string) and "domain" (string)
2. "reasoning": one sentence explaining the plan

Example:
{{
  "queries": [
    {{"query": "FIR filing procedure", "domain": "criminal_law"}},
    {{"query": "police complaint rights", "domain": "criminal_law"}}
  ],
  "reasoning": "User wants to file a police complaint; criminal_law domain is most relevant."
}}

Important: produce 1–3 queries. Return ONLY the JSON, nothing else.
"""


def _plan_research(
    user_message: str,
    conversation: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], str]:

    prompt = _PLAN_PROMPT.format(
        history=_format_history(conversation),
        user_message=user_message,
        domain_list=", ".join(list(KB_DOMAIN_IDS) + ["general"]),
    )
    raw = generate_explanation(prompt)
    if raw is None:
        return [{"query": user_message, "domain": "general"}], "Fallback — plan step failed."

    valid_domains = set(KB_DOMAIN_IDS) | {"general"}
    try:
        parsed = _extract_json_object(raw)
        queries = parsed.get("queries", [])
        if not isinstance(queries, list) or not queries:
            raise ValueError("Empty queries list")
        # Normalise domains
        for q in queries:
            if q.get("domain") not in valid_domains:
                q["domain"] = "general"
        return queries, parsed.get("reasoning", "")
    except Exception as exc:
        print(f"[planner] parse failed: {exc}\nRaw: {raw[:300]}")
        return [{"query": user_message, "domain": "general"}], "Fallback — plan parse failed."


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — QUERY REWRITER
# ─────────────────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """\
You are a search-query optimiser for an Indian legal knowledge base.

Original user message:
{user_message}

Planned queries:
{queries_json}

For each planned query, generate 1–2 alternative phrasings that capture the
same intent using different vocabulary (synonyms, legal terms, Hindi transliterations).

Return ONLY valid JSON — a list of objects, each with:
  "original": the original query string
  "rewrites": list of 1–2 alternative strings
  "domain": same domain as the original

Example:
[
  {{
    "original": "FIR filing procedure",
    "rewrites": ["how to register a First Information Report", "FIR darz karna"],
    "domain": "criminal"
  }}
]

Return ONLY the JSON array, nothing else.
"""


def _rewrite_queries(
    user_message: str,
    queries: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    
    prompt = _REWRITE_PROMPT.format(
        user_message=user_message,
        queries_json=json.dumps(queries, ensure_ascii=False, indent=2),
    )
    raw = generate_explanation(prompt)
    rewrite_log: List[Dict[str, Any]] = []
    expanded: List[Dict[str, str]] = list(queries)   # start with originals

    if raw is None:
        return expanded, rewrite_log

    try:
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, list):
            raise ValueError("Expected list")
        valid_domains = set(KB_DOMAIN_IDS) | {"general"}
        for item in parsed:
            original = item.get("original", "")
            domain = item.get("domain", "general")
            if domain not in valid_domains:
                domain = "general"
            rewrites = item.get("rewrites", [])
            rewrite_log.append({"original": original, "rewrites": rewrites})
            for r in rewrites:
                if r and r != original:
                    expanded.append({"query": r, "domain": domain})
    except Exception as exc:
        print(f"[rewriter] parse failed: {exc}")

    return expanded, rewrite_log


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SEARCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _run_planned_searches(
    queries: List[Dict[str, str]],
) -> Tuple[List[Passage], List[Dict[str, Any]]]:
    
    seen_ids: set = set()
    all_passages: List[Passage] = []
    trace: List[Dict[str, Any]] = []

    for q in queries:
        query_text = q.get("query", "")
        domain = q.get("domain", "general")
        results = _kb.search(query_text, domain=domain, top_k=TOP_K_PASSAGES)

        new_results = []
        for p in results:
            pid = p.metadata.get("id", id(p))
            if pid not in seen_ids:
                seen_ids.add(pid)
                new_results.append(p)

        all_passages.extend(new_results)
        trace.append({"query": query_text, "domain": domain, "found": len(new_results)})

        if len(all_passages) >= MAX_TOTAL_PASSAGES:
            break

    return all_passages[:MAX_TOTAL_PASSAGES], trace


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CHAT PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_prompt(
    conversation: List[Dict[str, str]],
    user_message: str,
    passages: List[Passage],
) -> str:
    context_blocks = []
    for i, p in enumerate(passages, 1):
        title = p.metadata.get("title", f"Source {i}")
        context_blocks.append(f"[{i}] {title}\n{p.page_content.strip()}")
    context = "\n\n".join(context_blocks)

    history = _format_history(conversation)

    return f"""\
You are Nyaya AI, a practical legal-awareness assistant for India.
You are NOT a lawyer, court, or government body.

KNOWLEDGE BASE CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER MESSAGE:
{user_message}

Instructions:
- Answer ONLY using the context above. Do not invent section numbers, helplines, or court names.
- Be practical and warm. Use simple everyday language.
- Give 3–5 short paragraphs: (1) what law/right applies, (2) immediate next steps,
  (3) evidence to preserve, (4) urgency note.
- If a helpline or portal appears in the context, cite it exactly.
- End with: "This is general educational information, not legal advice.
  Free legal aid is available through NALSA at 15100."
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SELF-CHECKER
# ─────────────────────────────────────────────────────────────────────────────

_SELF_CHECK_PROMPT = """\
You are a quality-control reviewer for Nyaya AI.

USER QUESTION:
{user_message}

DRAFT RESPONSE:
{draft}

KNOWLEDGE BASE PASSAGES USED:
{context}

Evaluate the draft. Return ONLY valid JSON:
{{
  "ok": true | false,
  "issues": ["issue 1", "issue 2"],
  "revised": "improved version of the response (only if ok=false, else empty string)",
  "gap_queries": [
    {{"query": "search query to fill a gap", "domain": "criminal"}}
  ]
}}

Rules:
- ok = true only if the draft is accurate, grounded in the passages, actionable, and answers the question.
- List every specific issue if ok = false.
- If information is missing, add gap_queries (max 2) to find it.
- In "revised", write the improved response (or empty string if ok = true).
- Return ONLY the JSON, nothing else.
"""


def _self_check_draft(
    user_message: str,
    draft: str,
    passages: List[Passage],
) -> Dict[str, Any]:
    context = "\n\n".join(
        f"[{i+1}] {p.metadata.get('title','')}\n{p.page_content[:400]}"
        for i, p in enumerate(passages)
    )
    prompt = _SELF_CHECK_PROMPT.format(
        user_message=user_message,
        draft=draft,
        context=context,
    )
    raw = generate_explanation(prompt)
    if raw is None:
        return {"ok": True, "issues": [], "revised": "", "gap_queries": []}

    try:
        parsed = _extract_json_object(raw)
        return {
            "ok": bool(parsed.get("ok", True)),
            "issues": parsed.get("issues", []),
            "revised": parsed.get("revised", ""),
            "gap_queries": parsed.get("gap_queries", []),
        }
    except Exception as exc:
        print(f"[self-check] parse failed: {exc}")
        return {"ok": True, "issues": [], "revised": "", "gap_queries": []}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — AGENT DECISION
# ─────────────────────────────────────────────────────────────────────────────

_DECISION_PROMPT = """\
You are the orchestration layer of Nyaya AI.

Conversation:
{history}

User message:
{user_message}

Planned searches:
{queries}

Choose exactly ONE action:
1. "answer"  — enough information is already in the conversation.
2. "clarify" — important facts are missing (state, timing, relationship, etc.).
3. "search"  — more knowledge-base information is needed.

Return ONLY valid JSON. Examples:

{{"action": "search", "questions": [], "reason": "Need to retrieve relevant passages."}}

{{"action": "clarify",
  "questions": ["Which state are you in?", "When did this happen?"],
  "reason": "Location and timing affect applicable law."}}

{{"action": "answer", "questions": [], "reason": "Prior context contains sufficient information."}}
"""


def _safe_parse_agent_decision(raw: str) -> Dict[str, Any]:
    try:
        parsed = _extract_json_object(raw)
    except Exception:
        return {"action": "search", "questions": [], "reason": "Parse failed."}

    action = parsed.get("action", "search")
    if action not in VALID_AGENT_ACTIONS:
        action = "search"

    return {
        "action": action,
        "questions": parsed.get("questions", []),
        "reason": parsed.get("reason", ""),
    }


def _decide_next_action(
    user_message: str,
    conversation: List[Dict[str, str]],
    planned_queries: List[Dict[str, str]],
) -> Dict[str, Any]:
    prompt = _DECISION_PROMPT.format(
        history=_format_history(conversation),
        user_message=user_message,
        queries=json.dumps(planned_queries, ensure_ascii=False, indent=2),
    )
    raw = generate_explanation(prompt)
    if raw is None:
        return {"action": "search", "questions": [], "reason": "Decision step failed."}
    return _safe_parse_agent_decision(raw)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — MAIN AGENTIC CHAT ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def agentic_chat_response(
    user_message: str,
    conversation: List[Dict[str, str]],
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
   
    trace: List[Dict[str, Any]] = []

    # ── 1. PLAN ──────────────────────────────────────────────────────────────
    queries, reasoning = _plan_research(user_message, conversation)

    if domain_hint and len(queries) == 1 and queries[0]["domain"] == "general":
        queries[0]["domain"] = domain_hint if domain_hint in KB_DOMAIN_IDS else "general"

    trace.append({
        "step": "plan",
        "label": "Planned research",
        "reasoning": reasoning,
        "queries": list(queries),
    })

    # ── 2. DECIDE ────────────────────────────────────────────────────────────
    decision = _decide_next_action(user_message, conversation, queries)
    action = decision["action"]

    trace.append({
        "step": "decide",
        "label": f"Decision: {action}",
        "reason": decision.get("reason", ""),
    })

    # ── 2a. CLARIFY branch ───────────────────────────────────────────────────
    if action == "clarify":
        questions = decision.get("questions") or [
            "Please provide more details so I can help accurately."
        ]
        trace.append({"step": "clarify", "label": "Asked clarifying questions", "questions": questions})
        return {
            "response": (
                "Before I can help accurately, please answer:\n\n• "
                + "\n• ".join(questions)
            ),
            "sources": [],
            "ai_available": True,
            "agent_trace": trace,
        }

    # ── 3. REWRITE ───────────────────────────────────────────────────────────
    expanded_queries, rewrite_log = _rewrite_queries(user_message, queries)
    trace.append({
        "step": "rewrite",
        "label": "Rewrote & expanded queries",
        "rewrites": rewrite_log,
        "expanded_queries": list(expanded_queries),
    })

    # ── 4. INITIAL SEARCH ────────────────────────────────────────────────────
    passages, search_trace = _run_planned_searches(expanded_queries)
    for s in search_trace:
        trace.append({
            "step": "search",
            "label": f"Searched: {s['query']}",
            "domain": s["domain"],
            "found": s["found"],
        })
    sources = _passages_to_sources(passages)

    if not passages:
        response_text = (
            "I couldn't find relevant information in the Nyaya AI knowledge base. "
            "For free legal aid, call NALSA: **15100**. "
            "For emergencies, call **112**."
        )
        trace.append({"step": "draft", "label": "No relevant material found", "skipped": True})
        return {"response": response_text, "sources": [], "ai_available": False, "agent_trace": trace}

    # ── 5. DRAFT → SELF-CHECK → GAP-FILL LOOP ───────────────────────────────
    seen_passage_ids: set = {p.metadata.get("id", id(p)) for p in passages}
    draft: Optional[str] = None
    check: Dict[str, Any] = {"ok": False, "issues": [], "revised": "", "gap_queries": []}

    for round_num in range(MAX_REFINEMENT_ROUNDS + 1):

        # 5a. Draft
        prompt = build_chat_prompt(conversation, user_message, passages)
        draft = generate_explanation(prompt)

        if draft is None:
            # Graceful degradation: show raw snippets
            snippets = "\n\n".join(
                f"**{p.metadata.get('title','')}**: {p.page_content[:250]}…"
                for p in passages[:2]
            )
            trace.append({"step": "draft", "label": "AI unavailable — used reference snippets", "skipped": True})
            return {
                "response": (
                    "I can't generate a full AI response right now. "
                    "Here's what our knowledge base says:\n\n"
                    + snippets
                    + "\n\nFor personalised help, call NALSA: **15100**."
                ),
                "sources": sources,
                "ai_available": False,
                "agent_trace": trace,
            }

        label = "Drafted response" if round_num == 0 else f"Re-drafted (round {round_num})"
        trace.append({"step": "draft", "label": label, "refinement_round": round_num})

        # 5b. Self-check
        check = _self_check_draft(user_message, draft, passages)
        trace.append({
            "step": "self_check",
            "label": "Self-check passed" if check["ok"] else (
                "Issues found — will refine" if round_num < MAX_REFINEMENT_ROUNDS
                else "Final self-check"
            ),
            "ok": check["ok"],
            "issues": check["issues"],
            "gap_queries": check["gap_queries"],
            "refinement_round": round_num,
        })

        # 5c. Stop if good or no gap queries remain
        if check["ok"] or not check["gap_queries"] or round_num == MAX_REFINEMENT_ROUNDS:
            break

        # 5d. Gap-fill search
        gap_passages, gap_trace = _run_planned_searches(check["gap_queries"])
        new_passages = [
            p for p in gap_passages
            if p.metadata.get("id", id(p)) not in seen_passage_ids
        ]
        for p in new_passages:
            seen_passage_ids.add(p.metadata.get("id", id(p)))

        for s in gap_trace:
            trace.append({
                "step": "gap_search",
                "label": f"Gap-fill: {s['query']}",
                "domain": s["domain"],
                "found": s["found"],
                "refinement_round": round_num,
            })

        if not new_passages:
            trace.append({"step": "gap_search", "label": "No new passages — stopping early", "refinement_round": round_num})
            break

        passages = (passages + new_passages)[:MAX_TOTAL_PASSAGES]
        sources = _passages_to_sources(passages)

    # Use revised text from self-checker if it improved the draft
    final_text = draft or UNAVAILABLE_MESSAGE
    if not check["ok"] and check.get("revised"):
        final_text = check["revised"]

    return {
        "response": final_text,
        "sources": sources,
        "ai_available": True,
        "agent_trace": trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — BASELINE (no retrieval)
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_PROMPT = """\
You are Nyaya AI, an educational civic-awareness assistant for India.
You are NOT a lawyer, court, or government authority.

A user described their situation (classified as: {primary_category}, risk level: {risk_level}):
\"\"\"{user_text}\"\"\"

Write a PRACTICAL, WARM response (3–5 short paragraphs) covering:
1. What this situation looks like legally — which law/right most likely applies.
2. Concrete next actions the person should take RIGHT NOW.
3. What evidence they should immediately preserve.
4. A note on urgency.

Rules:
- Be specific and actionable.
- Use simple, everyday language.
- End with one short sentence saying this is general educational information,
  not legal advice, and that free legal aid is available through NALSA (15100).
"""


def get_baseline_explanation(
    user_text: str,
    primary_category: str,
    risk_level: str,
) -> Dict[str, Any]:
    prompt = _BASELINE_PROMPT.format(
        user_text=user_text,
        primary_category=primary_category,
        risk_level=risk_level,
    )
    explanation = generate_explanation(prompt)
    if explanation is None:
        return {
            "ai_available": False,
            "ai_explanation": UNAVAILABLE_MESSAGE,
            "ai_sources": [],
            "self_check": {"ok": None, "issues": [], "revised": False},
        }
    return {
        "ai_available": True,
        "ai_explanation": explanation,
        "ai_sources": [],
        "self_check": {"ok": None, "issues": [], "revised": False},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — EVALUATOR (RAG vs Baseline)
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_RUBRIC = """\
You are an evaluation assistant for a legal-awareness AI research project.
Score two responses (RAG and Baseline) on four research dimensions.

USER QUERY:
\"\"\"{user_text}\"\"\"

REFERENCE MATERIAL (knowledge base passages used by RAG):
{context}

RAG RESPONSE:
\"\"\"{rag_response}\"\"\"

BASELINE RESPONSE (no retrieval context):
\"\"\"{baseline_response}\"\"\"

Score each dimension 1-5 for BOTH RAG and Baseline.

Dimensions:
1. GROUNDING     — Specific facts (helplines, sections, portals) traceable to reference material.
2. ACTIONABILITY — Concrete, immediately usable steps. (5=specific steps + helplines, 1=vague)
3. HALLUCINATION — Absence of invented facts. (5=zero hallucination, 1=major fabrications)
4. RELEVANCE     — Addresses the user's specific situation. (5=fully on-topic, 1=off-topic)

Respond with ONLY a JSON object:
{{"rag": {{"grounding": N, "actionability": N, "hallucination": N, "relevance": N}},
 "baseline": {{"grounding": N, "actionability": N, "hallucination": N, "relevance": N}},
 "rag_total": N, "baseline_total": N,
 "winner": "rag" | "baseline" | "tie",
 "key_difference": "one sentence on the main difference"}}
"""

_EVAL_RUBRIC_DEVIL = """\
You are a skeptical evaluator for a legal-awareness AI research project.
Apply these strict rules:
- GROUNDING: deduct 1 point for EACH specific fact (helpline, section, URL) not in reference material.
- HALLUCINATION: 5 means ZERO invented facts. Even one unverifiable helpline drops this to ≤3.
- ACTIONABILITY: vague advice like "consult a lawyer" without a helpline reduces this below 3.
- RELEVANCE: award 5 only if the response addresses the user's SPECIFIC situation.

USER QUERY:
\"\"\"{user_text}\"\"\"

REFERENCE MATERIAL:
{context}

RAG RESPONSE:
\"\"\"{rag_response}\"\"\"

BASELINE RESPONSE:
\"\"\"{baseline_response}\"\"\"

Respond with ONLY a JSON object:
{{"rag": {{"grounding": N, "actionability": N, "hallucination": N, "relevance": N}},
 "baseline": {{"grounding": N, "actionability": N, "hallucination": N, "relevance": N}},
 "rag_total": N, "baseline_total": N,
 "winner": "rag" | "baseline" | "tie",
 "key_difference": "one sentence from the skeptical viewpoint"}}
"""


def _call_gemini_judge(prompt_template: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        prompt_str = prompt_template.format(**kwargs)
    except Exception:
        return None
    raw = generate_explanation(prompt_str, temperature=0.0)
    if raw is None:
        return None
    try:
        return _extract_json_object(raw)
    except Exception:
        return None


# Cross-family judge constants
_CROSS_FAMILY_MODEL_OPENAI = os.environ.get("CROSS_FAMILY_MODEL_OPENAI", "gpt-4o")
_CROSS_FAMILY_MODEL_ANTHROPIC = os.environ.get("CROSS_FAMILY_MODEL_ANTHROPIC", "claude-sonnet-4-6")

_CROSS_FAMILY_RUBRIC = """\
You are an independent evaluation assistant for a legal-awareness AI research project.
Your scores MUST be independent of which response appears first.

USER QUERY:
\"\"\"{user_text}\"\"\"

REFERENCE MATERIAL (knowledge base passages used by the RAG system):
{context}

RESPONSE A:
\"\"\"{response_a}\"\"\"

RESPONSE B:
\"\"\"{response_b}\"\"\"

Score BOTH responses (A and B) 1–5 on each dimension:
1. GROUNDING     — Facts traceable to reference material. 5=fully grounded, 1=fabricated.
2. ACTIONABILITY — Concrete, immediately usable steps. 5=specific steps+helplines, 1=vague.
3. HALLUCINATION — Absence of invented facts. 5=zero hallucination, 1=major fabrications.
4. RELEVANCE     — Addresses the user's specific situation. 5=fully on-topic, 1=off-topic.

Respond with ONLY a JSON object:
{{"a": {{"grounding": N, "actionability": N, "hallucination": N, "relevance": N}},
 "b": {{"grounding": N, "actionability": N, "hallucination": N, "relevance": N}},
 "a_total": N, "b_total": N,
 "winner": "a" | "b" | "tie",
 "key_difference": "one sentence on the main difference"}}
"""


def _get_cross_family_judge_name() -> Optional[str]:
    if _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if _ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _call_cross_family_judge(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    family = _get_cross_family_judge_name()
    if family is None:
        return None

    prompt_text = _CROSS_FAMILY_RUBRIC.format(**kwargs)

    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            if family == "openai":
                client = _OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
                resp = client.chat.completions.create(
                    model=_CROSS_FAMILY_MODEL_OPENAI,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content or ""
            else:  # anthropic
                client = _anthropic_module.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                resp = client.messages.create(
                    model=_CROSS_FAMILY_MODEL_ANTHROPIC,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = resp.content[0].text if resp.content else ""

            return _extract_json_object(raw)
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_or_overload_error(exc):
                print(f"[judge] Cross-family judge ({family}) failed (non-retryable): {exc}")
                return None
            if attempt == GEMINI_MAX_RETRIES:
                break
            delay = _parse_retry_delay_seconds(exc)
            if delay is None:
                delay = min(GEMINI_BACKOFF_BASE_S * (2 ** attempt), GEMINI_BACKOFF_MAX_S)
            else:
                delay = min(delay + 0.5, GEMINI_BACKOFF_MAX_S)
            print(
                f"[judge] Cross-family judge ({family}) rate-limited/overloaded "
                f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES}); retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    print(f"[judge] Cross-family judge ({family}) failed after {GEMINI_MAX_RETRIES} retries: {last_exc}")
    return None


def evaluate_rag_vs_baseline(
    user_text: str,
    rag_response: str,
    baseline_response: str,
    passages: List[Passage],
    use_devil_advocate: bool = True,
    use_cross_family: bool = True,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite and return aggregated scores.
    """
    context = "\n\n".join(
        f"[{i+1}] {p.metadata.get('title','')}\n{p.page_content[:500]}"
        for i, p in enumerate(passages)
    )
    shared = dict(
        user_text=user_text,
        rag_response=rag_response,
        baseline_response=baseline_response,
        context=context,
    )

    # Standard judge (Gemini)
    standard = _call_gemini_judge(_EVAL_RUBRIC, shared)

    # Devil's advocate judge (Gemini)
    devil = _call_gemini_judge(_EVAL_RUBRIC_DEVIL, shared) if use_devil_advocate else None

    # Cross-family judge
    cross = None
    if use_cross_family:
        import random
        if random.random() > 0.5:
            cf_kwargs = dict(user_text=user_text, context=context,
                             response_a=rag_response, response_b=baseline_response)
            rag_is_a = True
        else:
            cf_kwargs = dict(user_text=user_text, context=context,
                             response_a=baseline_response, response_b=rag_response)
            rag_is_a = False

        raw_cross = _call_cross_family_judge(cf_kwargs)
        if raw_cross:
            # Remap a/b → rag/baseline
            if rag_is_a:
                cross = {"rag": raw_cross.get("a"), "baseline": raw_cross.get("b"),
                         "winner": {"a": "rag", "b": "baseline", "tie": "tie"}.get(raw_cross.get("winner","tie"), "tie"),
                         "key_difference": raw_cross.get("key_difference", "")}
            else:
                cross = {"rag": raw_cross.get("b"), "baseline": raw_cross.get("a"),
                         "winner": {"b": "rag", "a": "baseline", "tie": "tie"}.get(raw_cross.get("winner","tie"), "tie"),
                         "key_difference": raw_cross.get("key_difference", "")}

    return {"standard": standard, "devil": devil, "cross_family": cross}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — SAMPLE USAGE / QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "The police are refusing to register my FIR. What can I do?"
    print(f"\nQuery: {test_query}\n{'='*60}")

    result = agentic_chat_response(
        user_message=test_query,
        conversation=[],
        domain_hint="criminal_law",
    )

    print("RESPONSE:\n", result["response"])
    print("\nSOURCES:", result["sources"])
    print("\nAGENT TRACE STEPS:", [t["step"] for t in result["agent_trace"]])

def _blind_eval_kwargs(
    user_text: str,
    rag_response: str,
    baseline_response: str,
    context_str: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    if random.random() < 0.5:
        label_map = {"a": "rag", "b": "baseline"}
        response_a, response_b = rag_response, baseline_response
    else:
        label_map = {"a": "baseline", "b": "rag"}
        response_a, response_b = baseline_response, rag_response

    kwargs = {
        "user_text": user_text,
        "context": context_str,
        "response_a": response_a,
        "response_b": response_b,
    }
    return kwargs, label_map


def _deblind_cross_family_result(
    raw_result: Dict[str, Any],
    label_map: Dict[str, str],
) -> Dict[str, Any]:
   
    out: Dict[str, Any] = {}
    for label, system in label_map.items():
        if label in raw_result:
            out[system] = raw_result[label]

    # Totals
    inv = {v: k for k, v in label_map.items()}  # rag→a/b, baseline→a/b
    out["rag_total"] = raw_result.get(f"{inv.get('rag', 'a')}_total")
    out["baseline_total"] = raw_result.get(f"{inv.get('baseline', 'b')}_total")

    # Winner
    raw_winner = raw_result.get("winner", "tie")
    if raw_winner in label_map:
        out["winner"] = label_map[raw_winner]
    else:
        out["winner"] = "tie"

    out["key_difference"] = raw_result.get("key_difference", "")
    return out


def evaluate_rag_vs_baseline(
    user_text: str,
    rag_response: str,
    baseline_response: str,
    passages: Optional[List] = None,
    gold: Optional["GoldCase"] = None,
) -> dict:
   
    context_str = "\n\n".join(
    f"[{i+1}] {p.metadata.get('title','')}\n{p.page_content[:500]}"
    for i, p in enumerate(passages)
) if passages else "(no passages provided)"
 
    kwargs = dict(
    user_text=user_text,
    context=context_str,
    rag_response=rag_response,
    baseline_response=baseline_response,
)
    # ── Gemini judges ────────────────────────────
    j1 = _call_gemini_judge(_EVAL_RUBRIC, kwargs)
    j2 = _call_gemini_judge(_EVAL_RUBRIC_DEVIL, kwargs)

    # ── Cross-family judge  ────────────────────────────────────
    blind_kwargs, label_map = _blind_eval_kwargs(user_text, rag_response, baseline_response, context_str)
    j3_raw = _call_cross_family_judge(blind_kwargs)
    j3 = _deblind_cross_family_result(j3_raw, label_map) if j3_raw is not None else None
    cross_family_active = j3 is not None

    if j1 is None and j2 is None and j3 is None:
        return {
            "available": False,
            "error": "All judges unavailable (API error or missing keys)",
            "judge1": {}, "judge2": {}, "judge3": {},
            "kappa": {}, "rag": {}, "baseline": {},
            "cross_family_active": False,
        }


    first_available = next(j for j in (j1, j2, j3) if j is not None)
    if j1 is None:
        j1 = first_available
    if j2 is None:
        j2 = first_available

    # ── Per-dimension κ between judge pairs ────────────────────────────────
    def _pair_kappa(ja: Dict, jb: Dict, system: str) -> Dict[str, Optional[float]]:
        """Compute per-dim κ for a single judge pair on rag or baseline scores."""
        pair: Dict[str, Optional[float]] = {}
        a_scores = ja.get(system, {})
        b_scores = jb.get(system, {})
        for dim in EVAL_DIMS:
            s1, s2 = a_scores.get(dim), b_scores.get(dim)
            if s1 is not None and s2 is not None:
                pair[dim] = cohens_kappa([s1], [s2])
        return pair

    case_kappa: Dict[str, Any] = {
        "j1_vs_j2": {
            "rag": _pair_kappa(j1, j2, "rag"),
            "baseline": _pair_kappa(j1, j2, "baseline"),
        }
    }
    if cross_family_active:
        case_kappa["j1_vs_j3"] = {
            "rag": _pair_kappa(j1, j3, "rag"),
            "baseline": _pair_kappa(j1, j3, "baseline"),
        }
        case_kappa["j2_vs_j3"] = {
            "rag": _pair_kappa(j2, j3, "rag"),
            "baseline": _pair_kappa(j2, j3, "baseline"),
        }

    # ── Average scores across all active judges ─────────────────────────────
    active_judges = [j for j in (j1, j2, j3) if j is not None]

    def avg_scores(system: str) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for dim in EVAL_DIMS:
            vals = [j.get(system, {}).get(dim) for j in active_judges if j.get(system, {}).get(dim) is not None]
            out[dim] = round(sum(vals) / len(vals), 2) if vals else None
        return out

    avg_rag = avg_scores("rag")
    avg_baseline = avg_scores("baseline")
    rag_total = round(sum(v for v in avg_rag.values() if v is not None), 2)
    baseline_total = round(sum(v for v in avg_baseline.values() if v is not None), 2)

    # Majority winner across all active judges
    winner_votes: Dict[str, int] = {"rag": 0, "baseline": 0, "tie": 0}
    for j in active_judges:
        w = j.get("winner", "tie")
        winner_votes[w] = winner_votes.get(w, 0) + 1
    winner = max(winner_votes, key=lambda k: winner_votes[k])
    # If all three are different (impossible with 3 votes) or a genuine draw, call tie
    if winner_votes.get(winner, 0) == 1 and len(active_judges) == 3:
        winner = "tie"

    # ── Hallucination scores against gold (if provided) ────────────────────
    hall: Dict[str, Any] = {}
    if gold is not None:
        hall["rag"] = hallucination_scores(rag_response, gold)
        hall["baseline"] = hallucination_scores(baseline_response, gold)

    key_diff = (
        (j3 or j1).get("key_difference") or j2.get("key_difference") or ""
    )

    return {
        "available": True,
        "judge1": j1,
        "judge2": j2,
        "judge3": j3 if cross_family_active else None,
        "kappa": case_kappa,
        "rag": avg_rag,
        "baseline": avg_baseline,
        "rag_total": rag_total,
        "baseline_total": baseline_total,
        "winner": winner,
        "key_difference": key_diff,
        "hallucination": hall,
        "cross_family_active": cross_family_active,
        "cross_family_judge": _get_cross_family_judge_name(),
        "blind_label_map": label_map,
    }



def run_evaluation_suite(
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
  
    gold_lookup: Dict[str, GoldCase] = {g.text: g for g in GOLD_DATASET}

    results = []
    rag_totals: Dict[str, List[float]] = {d: [] for d in EVAL_DIMS}
    baseline_totals: Dict[str, List[float]] = {d: [] for d in EVAL_DIMS}

    # For aggregate κ — same-family (j1 vs j2) and cross-family (j1/j2 vs j3)
    judge1_all: List[Dict[str, int]] = []
    judge2_all: List[Dict[str, int]] = []
    judge3_all: List[Dict[str, int]] = []  # cross-family; may stay empty

    hall_rag_f1s: List[float] = []
    hall_base_f1s: List[float] = []

    wins = {"rag": 0, "baseline": 0, "tie": 0}
    any_cross_family = False

    for i, tc in enumerate(test_cases):
        gold = gold_lookup.get(tc["user_text"])

        passages = KnowledgeBaseRetriever(
            domain_ids=tc.get("domain_ids"), top_k=5
        ).invoke(tc["user_text"])

        rag_result = get_rag_explanation(
            user_text=tc["user_text"],
            primary_category=tc["primary_category"],
            risk_level=tc["risk_level"],
            domain_ids=tc.get("domain_ids", []),
        )
        baseline_result = get_baseline_explanation(
            user_text=tc["user_text"],
            primary_category=tc["primary_category"],
            risk_level=tc["risk_level"],
        )

        if not rag_result["ai_available"] or not baseline_result["ai_available"]:
            results.append({"case": i, "error": "AI unavailable", "user_text": tc["user_text"]})
            continue

        eval_result = evaluate_rag_vs_baseline(
            user_text=tc["user_text"],
            rag_response=rag_result["ai_explanation"],
            baseline_response=baseline_result["ai_explanation"],
            passages=passages,
            gold=gold,
        )

        case_record = {
            "case": i,
            "user_text": tc["user_text"][:120],
            "primary_category": tc["primary_category"],
            "risk_level": tc["risk_level"],
            "gold_matched": gold is not None,
            "gold_difficulty": gold.difficulty if gold else None,
            "rag_self_check": rag_result.get("self_check", {}),
            "eval": eval_result,
        }
        results.append(case_record)

        if eval_result.get("available"):
            for dim in EVAL_DIMS:
                rv = eval_result.get("rag", {}).get(dim)
                bv = eval_result.get("baseline", {}).get(dim)
                if rv is not None:
                    rag_totals[dim].append(rv)
                if bv is not None:
                    baseline_totals[dim].append(bv)

            winner = eval_result.get("winner", "tie")
            wins[winner] = wins.get(winner, 0) + 1

            # Same-family kappa accumulation (j1 vs j2)
            j1_rag = eval_result.get("judge1", {}).get("rag", {})
            j2_rag = eval_result.get("judge2", {}).get("rag", {})
            if j1_rag and j2_rag:
                judge1_all.append({d: j1_rag[d] for d in EVAL_DIMS if d in j1_rag})
                judge2_all.append({d: j2_rag[d] for d in EVAL_DIMS if d in j2_rag})

            # Cross-family kappa accumulation (j3)
            if eval_result.get("cross_family_active") and eval_result.get("judge3"):
                any_cross_family = True
                j3_rag = eval_result["judge3"].get("rag", {})
                if j3_rag:
                    judge3_all.append({d: j3_rag[d] for d in EVAL_DIMS if d in j3_rag})

            # Hallucination F1 (gold-matched cases only)
            hall = eval_result.get("hallucination", {})
            if gold and hall:
                rag_hl_f1 = hall.get("rag", {}).get("helplines", {}).get("f1")
                base_hl_f1 = hall.get("baseline", {}).get("helplines", {}).get("f1")
                if rag_hl_f1 is not None:
                    hall_rag_f1s.append(rag_hl_f1)
                if base_hl_f1 is not None:
                    hall_base_f1s.append(base_hl_f1)

    def _mean(lst: List[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 3) if lst else None

    # Same-family κ (j1 vs j2)
    kappa_same = kappa_across_cases(judge1_all, judge2_all)

    # Cross-family κ (j1 vs j3, j2 vs j3) — only if we have j3 scores
    kappa_cross_j1_j3: Dict[str, Optional[float]] = {}
    kappa_cross_j2_j3: Dict[str, Optional[float]] = {}
    if any_cross_family and judge3_all:
        min_len = min(len(judge1_all), len(judge3_all))
        kappa_cross_j1_j3 = kappa_across_cases(judge1_all[:min_len], judge3_all[:min_len])
        min_len2 = min(len(judge2_all), len(judge3_all))
        kappa_cross_j2_j3 = kappa_across_cases(judge2_all[:min_len2], judge3_all[:min_len2])

    aggregate = {
        "n_cases": len(test_cases),
        "n_evaluated": len([r for r in results if "eval" in r and r["eval"].get("available")]),
        "n_gold_matched": sum(1 for r in results if r.get("gold_matched")),
        "rag_means": {k: _mean(v) for k, v in rag_totals.items()},
        "baseline_means": {k: _mean(v) for k, v in baseline_totals.items()},
        "kappa_means": {
            "same_family_j1_vs_j2": kappa_same,
            "cross_family_j1_vs_j3": kappa_cross_j1_j3,
            "cross_family_j2_vs_j3": kappa_cross_j2_j3,
        },
        "hallucination_f1": {
            "rag_helpline_f1_mean": _mean(hall_rag_f1s),
            "baseline_helpline_f1_mean": _mean(hall_base_f1s),
            "n_gold_cases": len(hall_rag_f1s),
        },
        "wins": wins,
        "cross_family_active": any_cross_family,
        "cross_family_judge": _get_cross_family_judge_name(),
    }

    return {"results": results, "aggregate": aggregate}
