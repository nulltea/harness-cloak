"""Build a smaller common-entry lattice profile artifact.

This is a launch cleanup spike, not the canonical exhaustive lattice builder.
It keeps at most N entries per runtime type, removes conservative near
duplicates, and deliberately treats the existing profile ``count`` field only
as a source-local hint where the source actually encodes frequency/population.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from cloak.lattice.profiles import validate_profile_artifact

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

COMMON_HEALTH_EXACT = {
    "acne",
    "allergy",
    "alzheimer disease",
    "anemia",
    "anxiety disorder",
    "arthritis",
    "asthma",
    "atrial fibrillation",
    "autism spectrum disorder",
    "bronchitis",
    "cancer",
    "cataract",
    "chlamydia",
    "chronic obstructive pulmonary disease",
    "concussion",
    "coronary artery disease",
    "covid-19",
    "depression",
    "diabetes mellitus",
    "eczema",
    "epilepsy",
    "gastroenteritis",
    "glaucoma",
    "gonorrhea",
    "heart failure",
    "hepatitis",
    "hypertension",
    "influenza",
    "kidney stone",
    "leukemia",
    "migraine",
    "myocardial infarction",
    "obesity",
    "osteoarthritis",
    "osteoporosis",
    "pneumonia",
    "psoriasis",
    "schizophrenia",
    "stroke",
    "tuberculosis",
    "urinary tract infection",
}

COMMON_HEALTH_HEADS = {
    "acne",
    "allergy",
    "anemia",
    "anxiety",
    "arthritis",
    "asthma",
    "bronchitis",
    "cancer",
    "cataract",
    "chlamydia",
    "depression",
    "diabetes",
    "eczema",
    "epilepsy",
    "fracture",
    "glaucoma",
    "gonorrhea",
    "hepatitis",
    "hypertension",
    "infection",
    "influenza",
    "leukemia",
    "lymphoma",
    "migraine",
    "obesity",
    "pneumonia",
    "psoriasis",
    "stroke",
    "tuberculosis",
}

RARE_HEALTH_MARKERS = {
    "absolute",
    "achondrogenesis",
    "aciduria",
    "acquired",
    "acute",
    "adult",
    "autosomal",
    "congenital",
    "deficiency",
    "familial",
    "hereditary",
    "infantile",
    "juvenile",
    "malformation",
    "neoplasm",
    "pediatric",
    "primary",
    "rare",
    "secondary",
    "syndrome",
    "timothy",
    "type",
}

HEALTH_FAMILY_COLLAPSE_WORDS = {
    "allergy": "allergy",
    "allergic": "allergy",
    "cancer": "cancer",
    "carcinoma": "cancer",
    "leukemia": "leukemia",
    "lymphoma": "lymphoma",
    "melanoma": "melanoma",
    "neoplasm": "neoplasm",
    "sarcoma": "sarcoma",
}

HEALTH_FAMILY_PRIORITY = [
    "allergy",
    "allergic",
    "cancer",
    "carcinoma",
    "neoplasm",
    "sarcoma",
    "melanoma",
    "leukemia",
    "lymphoma",
    "diabetes",
    "hypertension",
    "hepatitis",
    "pneumonia",
    "tuberculosis",
    "asthma",
    "anemia",
    "arthritis",
    "depression",
    "epilepsy",
    "infection",
]

COMMON_DRUG_RANKED = [
    "atorvastatin",
    "metformin",
    "levothyroxine",
    "lisinopril",
    "amlodipine",
    "metoprolol",
    "albuterol",
    "losartan",
    "gabapentin",
    "omeprazole",
    "sertraline",
    "rosuvastatin",
    "pantoprazole",
    "escitalopram",
    "dextroamphetamine",
    "dextroamphetamine saccharate",
    "amphetamine",
    "amphetamine aspartate",
    "hydrochlorothiazide",
    "bupropion",
    "fluoxetine",
    "semaglutide",
    "montelukast",
    "trazodone",
    "simvastatin",
    "amoxicillin",
    "tamsulosin",
    "acetaminophen",
    "hydrocodone",
    "fluticasone",
    "meloxicam",
    "apixaban",
    "furosemide",
    "insulin glargine",
    "duloxetine",
    "ibuprofen",
    "famotidine",
    "empagliflozin",
    "carvedilol",
    "tramadol",
    "alprazolam",
    "prednisone",
    "hydroxyzine",
    "buspirone",
    "clopidogrel",
    "glipizide",
    "citalopram",
    "potassium chloride",
    "allopurinol",
    "aspirin",
    "cyclobenzaprine",
    "ergocalciferol",
    "oxycodone",
    "methylphenidate",
    "venlafaxine",
    "spironolactone",
    "ondansetron",
    "zolpidem",
    "cetirizine",
    "estradiol",
    "pravastatin",
    "lamotrigine",
    "quetiapine",
    "salmeterol",
    "clonazepam",
    "dulaglutide",
    "azithromycin",
    "clavulanate",
    "latanoprost",
    "cholecalciferol",
    "propranolol",
    "ezetimibe",
    "topiramate",
    "paroxetine",
    "diclofenac",
    "budesonide",
    "formoterol",
    "atenolol",
    "lisdexamfetamine",
    "doxycycline",
    "pregabalin",
    "ethinyl estradiol",
    "norethindrone",
    "glimepiride",
    "tizanidine",
    "clonidine",
    "fenofibrate",
    "insulin lispro",
    "valsartan",
    "cephalexin",
    "baclofen",
    "rivaroxaban",
    "ferrous sulfate",
    "amitriptyline",
    "finasteride",
    "dapagliflozin",
    "folic acid",
    "aripiprazole",
    "olmesartan",
    "norgestimate",
    "valacyclovir",
    "mirtazapine",
    "lorazepam",
    "levetiracetam",
    "insulin aspart",
    "naproxen",
    "cyanocobalamin",
    "loratadine",
    "diltiazem",
    "sumatriptan",
    "triamcinolone",
    "hydralazine",
    "tirzepatide",
    "celecoxib",
    "alendronate",
    "oxybutynin",
    "triamterene",
    "warfarin",
    "progesterone",
    "umeclidinium",
    "vilanterol",
    "testosterone",
    "nifedipine",
    "methocarbamol",
    "benzonatate",
    "sitagliptin",
    "chlorthalidone",
    "isosorbide",
    "donepezil",
    "dexmethylphenidate",
    "sulfamethoxazole",
    "trimethoprim",
    "clobetasol",
    "methotrexate",
    "hydroxychloroquine",
    "lovastatin",
    "pioglitazone",
    "irbesartan",
    "methylprednisolone",
    "meclizine",
    "levonorgestrel",
    "ketoconazole",
    "thyroid",
    "azelastine",
    "nitrofurantoin",
    "adalimumab",
    "memantine",
    "prednisolone",
    "esomeprazole",
    "docusate",
    "clindamycin",
    "acyclovir",
    "sildenafil",
    "insulin degludec",
    "insulin detemir",
    "drospirenone",
    "ciprofloxacin",
    "morphine",
    "insulin human",
    "insulin isophane human",
    "levocetirizine",
    "nirmatrelvir",
    "ritonavir",
    "valproate",
    "atomoxetine",
    "tiotropium",
    "melatonin",
    "cefdinir",
    "doxepin",
    "olanzapine",
    "phentermine",
    "ofloxacin",
    "etonogestrel",
    "mupirocin",
    "benazepril",
    "timolol",
    "magnesium salts",
    "fluconazole",
    "risperidone",
    "verapamil",
    "linaclotide",
    "cyclosporine",
    "doxazosin",
    "ipratropium",
    "hydrocortisone",
    "diazepam",
    "telmisartan",
    "carbamazepine",
    "lithium",
    "evolocumab",
    "desvenlafaxine",
    "dorzolamide",
    "nebivolol",
    "dicyclomine",
    "torsemide",
    "anastrozole",
    "enalapril",
    "polyethylene glycol 3350",
    "tretinoin",
    "tadalafil",
    "sacubitril",
    "calcium",
    "pramipexole",
    "mesalamine",
    "metronidazole",
    "nortriptyline",
    "emtricitabine",
    "tenofovir",
    "rimegepant",
    "nitroglycerin",
    "rizatriptan",
    "liraglutide",
    "codeine",
    "ramipril",
    "ropinirole",
    "brimonidine",
    "mirabegron",
    "colchicine",
    "ticagrelor",
    "terazosin",
    "amiodarone",
    "fexofenadine",
    "liothyronine",
    "bisoprolol",
    "omega-3-acid ethyl esters",
    "flecainide",
    "oxcarbazepine",
    "desogestrel",
    "ascorbic acid",
    "sodium salts",
    "ketorolac",
    "promethazine",
    "levofloxacin",
    "labetalol",
    "nystatin",
    "cyproheptadine",
    "erythromycin",
    "dutasteride",
    "moxifloxacin",
    "bimatoprost",
    "primidone",
    "sucralfate",
    "betamethasone",
    "clotrimazole",
    "senna",
    "bumetanide",
    "icosapent ethyl",
    "solifenacin",
    "dexamethasone",
    "epinephrine",
    "penicillin v",
    "calcitriol",
    "oseltamivir",
    "polymyxin b",
    "dextromethorphan",
    "terbinafine",
    "linagliptin",
    "methimazole",
    "metoclopramide",
    "medroxyprogesterone",
    "pancrelipase",
    "neomycin",
    "calcium phosphate",
    "butalbital",
    "caffeine",
    "guanfacine",
    "sodium fluoride",
    "guaifenesin",
    "lactulose",
    "fluorouracil",
    "olopatadine",
    "chlorhexidine",
    "nabumetone",
    "mometasone",
    "polyethylene glycol 3350 with electrolytes",
    "hydroquinone",
    "phenazopyridine",
    "loperamide",
    "lidocaine",
    "ciclopirox",
    "cefuroxime",
    "brompheniramine",
    "pseudoephedrine",
    "norgestrel",
    "diphenhydramine",
    "norelgestromin",
    "atropine",
    "diphenoxylate",
    "indomethacin",
    "niacin",
    "lactate",
    "vitamin e",
    "bisacodyl",
    "riboflavin",
    "ivermectin",
    "etodolac",
    "lactobacillus acidophilus",
    "tobramycin",
    "ketotifen",
]
COMMON_DRUG_RANK = {name: rank for rank, name in enumerate(COMMON_DRUG_RANKED, start=1)}
COMMON_DRUG_EXACT = set(COMMON_DRUG_RANKED)
DRUG_SOURCE_ID_LIMIT = 5

CURATED_DRUG_ALIASES = {
    "acetaminophen": {"apap", "paracetamol", "tylenol"},
    "albuterol": {"salbutamol"},
    "epinephrine": {"adrenaline"},
    "lidocaine": {"lignocaine"},
}

DRUG_ALIAS_BRAND_TOKENS = {
    "acetaminophen": {"tylenol"},
    "ibuprofen": {"advil", "motrin"},
    "omeprazole": {"prilosec"},
}

DRUG_ALIAS_ALLOWED_VERSION_WORDS = {
    "base",
    "basis",
    "caplet",
    "caplets",
    "capsule",
    "capsules",
    "chewable",
    "chewables",
    "child",
    "children",
    "childrens",
    "coated",
    "delayed",
    "dr",
    "dye",
    "extended",
    "extra",
    "film",
    "free",
    "gelcap",
    "gelcaps",
    "infant",
    "infants",
    "micronized",
    "oral",
    "pediatric",
    "red",
    "regular",
    "release",
    "strength",
    "suspension",
    "tablet",
    "tablets",
}

DRUG_ALIAS_REJECT_WORDS = {
    "allergy",
    "cold",
    "congestion",
    "cough",
    "flu",
    "headache",
    "migraine",
    "multi",
    "multisymptom",
    "night",
    "nighttime",
    "pm",
    "sinus",
    "sleep",
    "severe",
}

DRUG_FORM_WORDS = {
    "acetate",
    "anhydrous",
    "amylase",
    "aspartate",
    "aqueous",
    "bromide",
    "calcium",
    "capsule",
    "chloride",
    "cream",
    "dihydrochloride",
    "extended",
    "hcl",
    "hydrochloride",
    "hydrate",
    "injection",
    "liquid",
    "ointment",
    "potassium",
    "release",
    "sodium",
    "solution",
    "saccharate",
    "sulfate",
    "tablet",
    "topical",
}

COMMON_PROFESSION_EXACT = {
    "accountant",
    "architect",
    "artist",
    "attorney",
    "barber",
    "bartender",
    "cashier",
    "chef",
    "clerk",
    "cook",
    "counselor",
    "dentist",
    "designer",
    "doctor",
    "driver",
    "electrician",
    "engineer",
    "farmer",
    "firefighter",
    "janitor",
    "journalist",
    "lawyer",
    "manager",
    "mechanic",
    "nurse",
    "pharmacist",
    "photographer",
    "physician",
    "pilot",
    "plumber",
    "police officer",
    "professor",
    "programmer",
    "receptionist",
    "reporter",
    "salesperson",
    "scientist",
    "secretary",
    "security guard",
    "software developer",
    "teacher",
    "therapist",
    "veterinarian",
    "waiter",
    "writer",
}

COMMON_PROFESSION_HEADS = {
    "accountant",
    "administrator",
    "aide",
    "analyst",
    "artist",
    "assistant",
    "attorney",
    "cashier",
    "chef",
    "clerk",
    "cook",
    "counselor",
    "developer",
    "director",
    "doctor",
    "driver",
    "engineer",
    "guard",
    "janitor",
    "lawyer",
    "manager",
    "mechanic",
    "nurse",
    "operator",
    "pharmacist",
    "physician",
    "programmer",
    "representative",
    "salesperson",
    "scientist",
    "secretary",
    "specialist",
    "supervisor",
    "teacher",
    "technician",
    "therapist",
    "worker",
    "writer",
}

RARE_PROFESSION_MARKERS = {
    "abalone",
    "abrasive",
    "assistant to",
    "deputy",
    "intern",
    "junior",
    "senior",
    "trainee",
}

COMMON_PROCEDURE_EXACT = {
    "acupuncture",
    "angioplasty",
    "blood transfusion",
    "chemotherapy",
    "colonoscopy",
    "dialysis",
    "echocardiography",
    "electrocardiogram",
    "endoscopy",
    "fluoroscopy",
    "magnetic resonance imaging",
    "mammography",
    "physical therapy",
    "radiography",
    "radiotherapy",
    "ultrasound",
    "x-ray",
}

COMMON_PROCEDURE_CONCEPT_RANKED = [
    "appendectomy",
    "cesarean_section",
    "circumcision",
    "cholecystectomy",
    "colectomy",
    "hysterectomy",
    "colonoscopy",
    "upper_endoscopy",
    "cystoscopy",
    "bronchoscopy",
    "lens_extraction",
    "knee_replacement",
    "hip_replacement",
    "spinal_fusion",
    "coronary_angioplasty",
    "coronary_bypass",
    "ct_scan",
    "mri",
    "chest_xray",
    "ultrasound",
    "mammography",
    "fluoroscopy_upper_gi",
    "electrocardiogram",
    "echocardiography",
    "electroconvulsive_therapy",
    "abortion",
    "drainage",
    "occlusion",
    "extirpation",
    "dilation",
    "insertion",
    "removal",
    "repair",
    "replacement",
    "destruction",
    "fragmentation",
    "measurement",
    "monitoring",
    "control",
    "immobilization",
    "compression",
    "traction",
    "acupuncture",
]
COMMON_PROCEDURE_CONCEPT_RANK = {
    concept: rank
    for rank, concept in enumerate(COMMON_PROCEDURE_CONCEPT_RANKED, start=1)
}

PROCEDURE_PUBLIC_SURFACES = {
    "appendectomy": "appendectomy",
    "cesarean_section": "cesarean section",
    "circumcision": "circumcision",
    "cholecystectomy": "cholecystectomy",
    "colectomy": "colectomy",
    "hysterectomy": "hysterectomy",
    "colonoscopy": "colonoscopy",
    "upper_endoscopy": "upper endoscopy",
    "cystoscopy": "cystoscopy",
    "bronchoscopy": "bronchoscopy",
    "lens_extraction": "cataract surgery",
    "knee_replacement": "knee replacement",
    "hip_replacement": "hip replacement",
    "spinal_fusion": "spinal fusion",
    "coronary_angioplasty": "coronary angioplasty",
    "coronary_bypass": "coronary bypass surgery",
    "ct_scan": "ct scan",
    "mri": "mri",
    "chest_xray": "chest x-ray",
    "ultrasound": "ultrasound",
    "mammography": "mammography",
    "fluoroscopy_upper_gi": "upper gi fluoroscopy",
    "electrocardiogram": "electrocardiogram",
    "echocardiography": "echocardiography",
    "electroconvulsive_therapy": "electroconvulsive therapy",
    "abortion": "abortion",
    "drainage": "drainage",
    "occlusion": "occlusion",
    "extirpation": "extirpation",
    "dilation": "dilation",
    "insertion": "insertion",
    "removal": "removal",
    "repair": "repair",
    "replacement": "replacement",
    "destruction": "destruction",
    "fragmentation": "fragmentation",
    "measurement": "physiologic test",
    "monitoring": "clinical observation",
    "control": "control",
    "immobilization": "immobilization",
    "compression": "compression",
    "traction": "traction",
    "acupuncture": "acupuncture",
}

PROCEDURE_CONCEPT_PREFERRED_TOKENS = {
    "ct_scan": {"abdomen"},
    "mri": {"brain"},
    "chest_xray": {"chest"},
    "ultrasound": {"abdomen"},
    "coronary_angioplasty": {"percutaneous"},
    "colonoscopy": {"endoscopic"},
    "upper_endoscopy": {"endoscopic"},
    "cystoscopy": {"natural"},
    "bronchoscopy": {"endoscopic"},
    "cesarean_section": {"low"},
}

COMMON_PROCEDURE_HEADS = {
    "assessment",
    "biopsy",
    "catheterization",
    "chemotherapy",
    "colonoscopy",
    "dialysis",
    "drainage",
    "echocardiography",
    "endoscopy",
    "excision",
    "fluoroscopy",
    "imaging",
    "inspection",
    "measurement",
    "monitoring",
    "radiography",
    "resection",
    "therapy",
    "tomography",
    "transfusion",
    "ultrasonography",
    "ultrasound",
}

PROCEDURE_DETAIL_WORDS = {
    "approach",
    "bilateral",
    "contrast",
    "endoscopic",
    "external",
    "guidance",
    "high",
    "left",
    "low",
    "multiple",
    "open",
    "other",
    "percutaneous",
    "right",
    "single",
    "substitute",
    "synthetic",
    "using",
    "with",
}

COMMON_ORG_WORDS = {
    "ambulance",
    "care",
    "center",
    "clinic",
    "community",
    "dental",
    "diagnostic",
    "family",
    "health",
    "home",
    "hospice",
    "hospital",
    "laboratory",
    "medical",
    "pharmacy",
    "rehab",
    "rehabilitation",
    "urgent",
    "wellness",
}

LEGAL_SUFFIX_WORDS = {
    "co",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "lp",
    "ltd",
    "pa",
    "pc",
    "pllc",
}


def norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(norm(text))


def score_row(runtime_type: str, surface: str, row: dict[str, Any]) -> float:
    ts = tokens(surface)
    tset = set(ts)
    score = 0.0

    if runtime_type == "LOC":
        source_count = float(row.get("count", 1.0) or 1.0)
        score += math.log1p(max(source_count, 1.0))
        if any(str(s).startswith("geonames-country:") for s in row.get("source_ids", [])):
            score += 3.0
        return score - 0.05 * len(ts)

    if runtime_type == "organization-medical-facility":
        score += 4.0 * len(tset & COMMON_ORG_WORDS)
        score += 0.4 * math.log1p(float(row.get("count", 1.0) or 1.0))
        score -= 0.7 * len(tset & LEGAL_SUFFIX_WORDS)
        score -= 0.25 * max(len(ts) - 4, 0)
        return score

    if runtime_type == "drug":
        rank = COMMON_DRUG_RANK.get(surface)
        if rank is None:
            return -1.0
        score += 1000.0 - float(rank)
        score += 0.3 * math.log1p(float(row.get("count", 1.0) or 1.0))
        score -= 0.7 * len(tset & DRUG_FORM_WORDS)
        score -= 1.3 * max(len(ts) - 2, 0)
        if any(ch in surface for ch in "()."):
            score -= 2.0
        if "nosode" in tset or "homeopathic" in tset:
            score -= 5.0
        return score

    if runtime_type == "health-condition":
        if surface in COMMON_HEALTH_EXACT:
            score += 25.0
        score += 4.0 * len(tset & COMMON_HEALTH_HEADS)
        score -= 2.5 * len(tset & RARE_HEALTH_MARKERS)
        score -= 1.2 * max(len(ts) - 3, 0)
        if any(ch.isdigit() for ch in surface):
            score -= 3.0
        if "-" in surface:
            score -= 0.8
        return score

    if runtime_type == "profession":
        if surface in COMMON_PROFESSION_EXACT:
            score += 20.0
        score += 4.0 * len(tset & COMMON_PROFESSION_HEADS)
        if any(str(s).startswith(("onet-job-title:", "onet:", "isco:")) for s in row.get("source_ids", [])):
            score += 1.0
        if any(marker in surface for marker in RARE_PROFESSION_MARKERS):
            score -= 2.5
        score -= 0.8 * max(len(ts) - 2, 0)
        return score

    if runtime_type == "medical-procedure":
        concept = procedure_concept(surface)
        if concept is None:
            return -1.0
        score += 1000.0 - float(COMMON_PROCEDURE_CONCEPT_RANK[concept])
        score += 2.0 * len(tset & PROCEDURE_CONCEPT_PREFERRED_TOKENS.get(concept, set()))
        if concept == "coronary_angioplasty" and "open" in tset:
            score -= 3.0
        elif concept == "cystoscopy" and "open" in tset:
            score -= 2.0
        elif "open" in tset:
            score += 0.5
        score -= 0.3 * len(tset & PROCEDURE_DETAIL_WORDS)
        score -= 0.05 * max(len(ts) - 5, 0)
        return score

    return 1.0


def procedure_concept(surface: str) -> str | None:
    s = norm(surface)
    ts = set(tokens(s))
    if s == "acupuncture":
        return "acupuncture"
    if s.startswith("abortion of products of conception"):
        return "abortion"
    if "extraction" in ts and {"products", "conception"} <= ts:
        return "cesarean_section"
    for root in (
        "drainage",
        "occlusion",
        "dilation",
        "insertion",
        "removal",
        "repair",
        "replacement",
        "destruction",
        "measurement",
        "monitoring",
        "control",
        "immobilization",
        "compression",
        "traction",
    ):
        if s.startswith(root):
            return root
    if s.startswith("extirpation of matter"):
        return "extirpation"
    if s.startswith("fragmentation in"):
        return "fragmentation"
    if "prepuce" in ts and ts & {"excision", "resection"}:
        return "circumcision"
    if "appendix" in ts and ts & {"excision", "resection", "extraction"}:
        return "appendectomy"
    if "gallbladder" in ts and ts & {"excision", "resection", "extraction"}:
        return "cholecystectomy"
    if "colon" in ts and ts & {"excision", "resection"}:
        return "colectomy"
    if "uterus" in ts and ts & {"excision", "resection"}:
        return "hysterectomy"
    if "inspection" in ts and {"lower", "intestinal", "tract"} <= ts:
        return "colonoscopy"
    if "inspection" in ts and (
        {"upper", "intestinal", "tract"} <= ts
        or "stomach" in ts
    ):
        return "upper_endoscopy"
    if "inspection" in ts and "bladder" in ts:
        return "cystoscopy"
    if "inspection" in ts and "tracheobronchial" in ts:
        return "bronchoscopy"
    if "lens" in ts and "extraction" in ts:
        return "lens_extraction"
    if "knee" in ts and "replacement" in ts:
        return "knee_replacement"
    if "hip" in ts and "replacement" in ts:
        return "hip_replacement"
    if "fusion" in ts and ("spine" in ts or "vertebral" in ts):
        return "spinal_fusion"
    if "dilation" in ts and "coronary" in ts and "artery" in ts:
        return "coronary_angioplasty"
    if "bypass" in ts and "coronary" in ts and "artery" in ts:
        return "coronary_bypass"
    if s.startswith("computerized tomography (ct scan)"):
        return "ct_scan"
    if s.startswith("magnetic resonance imaging (mri)"):
        return "mri"
    if s.startswith("plain radiography") and "chest" in ts:
        return "chest_xray"
    if s.startswith("ultrasonography"):
        return "ultrasound"
    if "mammography" in ts or (s.startswith("plain radiography") and "breasts" in ts):
        return "mammography"
    if s == "fluoroscopy of upper gi":
        return "fluoroscopy_upper_gi"
    if "electrocardiogram" in ts:
        return "electrocardiogram"
    if "echocardiography" in ts:
        return "echocardiography"
    if "electroconvulsive" in ts and "therapy" in ts:
        return "electroconvulsive_therapy"
    return None


def dedupe_key(runtime_type: str, surface: str) -> str:
    ts = tokens(surface)
    if runtime_type == "health-condition":
        for word in HEALTH_FAMILY_PRIORITY:
            if word in ts:
                return HEALTH_FAMILY_COLLAPSE_WORDS.get(word, word)
        for word in sorted(COMMON_HEALTH_HEADS):
            if word in ts:
                return word
        ts = [t for t in ts if t not in RARE_HEALTH_MARKERS]
    elif runtime_type == "drug":
        ts = [t for t in ts if t not in DRUG_FORM_WORDS]
    elif runtime_type == "medical-procedure":
        concept = procedure_concept(surface)
        if concept:
            return concept
        ts = [t for t in ts if t not in PROCEDURE_DETAIL_WORDS]
    elif runtime_type == "organization-medical-facility":
        ts = [t for t in ts if t not in LEGAL_SUFFIX_WORDS]
    return " ".join(ts) or norm(surface)


def is_near_duplicate(runtime_type: str, surface: str, row: dict[str, Any], kept: dict[str, dict[str, Any]]) -> bool:
    if runtime_type in {"LOC", "nationality", "religion", "gender", "marital-status", "sexual-orientation", "ORG"}:
        return False

    st = set(tokens(surface))
    if not st:
        return False
    levels = tuple(row.get("levels", []))
    exact_protected = (
        surface in COMMON_HEALTH_EXACT
        or surface in COMMON_DRUG_EXACT
        or surface in COMMON_PROFESSION_EXACT
        or surface in COMMON_PROCEDURE_EXACT
    )
    if exact_protected:
        return False

    for kept_surface, kept_row in kept.items():
        kt = set(tokens(kept_surface))
        if not kt or not (kt < st or st < kt):
            continue
        extra = st - kt if kt < st else kt - st
        same_levels = levels == tuple(kept_row.get("levels", []))
        if runtime_type == "health-condition" and same_levels:
            return True
        if runtime_type == "drug" and (extra <= DRUG_FORM_WORDS or same_levels):
            return True
        if runtime_type == "medical-procedure" and (extra <= PROCEDURE_DETAIL_WORDS or same_levels):
            return True
        if runtime_type == "profession" and same_levels and (extra & {"senior", "junior", "assistant", "deputy", "trainee"}):
            return True
        if runtime_type == "organization-medical-facility" and extra <= LEGAL_SUFFIX_WORDS:
            return True
    return False


def select_common_entries(runtime_type: str, entries: dict[str, dict[str, Any]], limit: int) -> dict[str, dict[str, Any]]:
    best_by_key: dict[str, tuple[float, str, dict[str, Any]]] = {}
    for surface, row in entries.items():
        score = score_row(runtime_type, surface, row)
        if _below_commonness_floor(runtime_type, score, surface):
            continue
        key = dedupe_key(runtime_type, surface)
        cur = best_by_key.get(key)
        challenger = (score, surface, row)
        if cur is None or _rank_tuple(challenger) > _rank_tuple(cur):
            best_by_key[key] = challenger

    ranked = sorted(best_by_key.values(), key=lambda item: (-item[0], len(tokens(item[1])), item[1]))
    selected: dict[str, dict[str, Any]] = {}
    for _score, surface, row in ranked:
        if is_near_duplicate(runtime_type, surface, row, selected):
            continue
        public_surface = public_profile_surface(runtime_type, surface)
        selected[public_surface] = clean_profile_row(runtime_type, public_surface, row, original_surface=surface)
        if len(selected) >= limit:
            break
    return dict(sorted(selected.items()))


def public_profile_surface(runtime_type: str, surface: str) -> str:
    if runtime_type == "medical-procedure":
        concept = procedure_concept(surface)
        if concept:
            return PROCEDURE_PUBLIC_SURFACES[concept]
    return surface


def clean_profile_row(
    runtime_type: str,
    surface: str,
    row: dict[str, Any],
    *,
    original_surface: str | None = None,
) -> dict[str, Any]:
    cleaned = {
        "aliases": list(row.get("aliases", [])),
        "levels": list(row.get("levels", [])),
        "source_ids": list(row.get("source_ids", [])),
        "count": row.get("count", 1.0),
    }
    if runtime_type == "drug":
        cleaned["source_ids"] = sorted(set(cleaned["source_ids"]))[:DRUG_SOURCE_ID_LIMIT]
        aliases = []
        for alias in sorted(CURATED_DRUG_ALIASES.get(surface, set())):
            if alias != surface:
                aliases.append(alias)
        for alias in cleaned["aliases"]:
            alias = norm(alias)
            if _is_clean_drug_alias(surface, alias) and alias not in aliases:
                aliases.append(alias)
        cleaned["aliases"] = sorted(aliases)
    elif runtime_type == "medical-procedure":
        aliases = []
        original_surface = norm(original_surface or "")
        if original_surface and original_surface != surface:
            aliases.append(original_surface)
        cleaned["aliases"] = aliases
    return cleaned


def _is_clean_drug_alias(surface: str, alias: str) -> bool:
    if not alias or alias == surface:
        return False
    if alias in CURATED_DRUG_ALIASES.get(surface, set()):
        return True

    alias_tokens = set(tokens(alias))
    if not alias_tokens or alias_tokens & DRUG_ALIAS_REJECT_WORDS:
        return False

    surface_tokens = set(tokens(surface))
    curated_tokens = set().union(*(set(tokens(a)) for a in CURATED_DRUG_ALIASES.get(surface, set()))) if CURATED_DRUG_ALIASES.get(surface) else set()
    allowed_extra = DRUG_FORM_WORDS | DRUG_ALIAS_ALLOWED_VERSION_WORDS | curated_tokens

    if surface_tokens and surface_tokens <= alias_tokens:
        extra = alias_tokens - surface_tokens
        if _mentions_other_common_drug(surface, alias_tokens):
            return False
        return len(alias_tokens) <= 7 and extra <= allowed_extra

    brand_tokens = DRUG_ALIAS_BRAND_TOKENS.get(surface, set())
    if brand_tokens and alias_tokens & brand_tokens:
        extra = alias_tokens - brand_tokens
        return len(alias_tokens) <= 5 and extra <= (DRUG_ALIAS_ALLOWED_VERSION_WORDS | {"children", "childrens"})

    return False


def _mentions_other_common_drug(surface: str, alias_tokens: set[str]) -> bool:
    for drug in COMMON_DRUG_EXACT:
        if drug == surface:
            continue
        drug_tokens = set(tokens(drug))
        if not drug_tokens or drug_tokens <= DRUG_FORM_WORDS:
            continue
        if drug_tokens <= alias_tokens:
            return True
    return False


def _below_commonness_floor(runtime_type: str, score: float, surface: str) -> bool:
    if runtime_type == "health-condition":
        return surface not in COMMON_HEALTH_EXACT
    if runtime_type == "drug":
        return surface not in COMMON_DRUG_EXACT
    if runtime_type == "medical-procedure":
        return procedure_concept(surface) is None
    if runtime_type == "profession":
        return score < -1.0 and surface not in COMMON_PROFESSION_EXACT
    return False


def _rank_tuple(item: tuple[float, str, dict[str, Any]]) -> tuple[float, float, str]:
    score, surface, row = item
    return (score, -float(len(tokens(surface))), surface)


def build_common_artifact(source: dict[str, Any], limit: int) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = {}
    report = {"limit_per_category": limit, "categories": {}}
    for runtime_type, entries in sorted(source.get("profiles", {}).items()):
        selected = select_common_entries(runtime_type, entries, limit)
        profiles[runtime_type] = selected
        report["categories"][runtime_type] = {
            "input": len(entries),
            "output": len(selected),
            "removed": len(entries) - len(selected),
        }

    artifact = {
        "schema_version": source.get("schema_version", 1),
        "created": str(date.today()),
        "sources": {
            "base": "data/lattice_profiles/fine_lattice_profiles.json",
            "cleanup": "scripts/spikes/build_common_lattice_profiles.py",
            "drug_commonness": "ClinCalc DrugStats Top 300 Drugs of 2023, derived from AHRQ MEPS",
            "selection": (
                "category-specific source/manual commonness scoring with conservative "
                "near-duplicate collapse; at most the requested limit per category"
            ),
        },
        "profiles": profiles,
    }
    return artifact, report


def duplicate_summary(artifact: dict[str, Any]) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = defaultdict(list)
    for runtime_type, entries in artifact.get("profiles", {}).items():
        buckets: dict[str, list[str]] = defaultdict(list)
        for surface in entries:
            buckets[dedupe_key(runtime_type, surface)].append(surface)
        for surfaces in buckets.values():
            if len(surfaces) > 1:
                out[runtime_type].append(sorted(surfaces))
    return {rt: groups[:20] for rt, groups in sorted(out.items()) if groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="data/lattice_profiles/fine_lattice_profiles.json")
    parser.add_argument("--out", default="data/lattice_profiles/comm_lattice_profiles.json")
    parser.add_argument("--report-out", default="results/common_lattice_profiles_report.json")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    source = json.loads(Path(args.in_path).read_text())
    artifact, report = build_common_artifact(source, args.limit)
    errors = validate_profile_artifact(artifact)
    if errors:
        raise SystemExit("invalid common lattice profile artifact:\n" + "\n".join(errors[:50]))

    report["near_duplicate_buckets_after_cleanup"] = duplicate_summary(artifact)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True))

    total = sum(len(v) for v in artifact["profiles"].values())
    print(f"wrote {total} profiles -> {out}", flush=True)
    print(f"report -> {report_out}", flush=True)


if __name__ == "__main__":
    main()
