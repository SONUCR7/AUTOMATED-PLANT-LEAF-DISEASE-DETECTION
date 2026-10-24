"""
Agronomic Advisory & Treatment Recommendation Engine
=====================================================
Maps detected plant disease classes and quantitative severity percentages
to actionable, evidence-based agricultural recommendations (organic, chemical,
and cultural practices).
"""

REMEDIES_DATABASE = {
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "symptoms": "Concentric rings ('target-board' or 'bullseye' pattern) with chlorotic yellow halo on older foliage, turning dark brown and necrotic.",
        "early_stage_advisory": "Remove and incinerate lower infected foliage. Avoid overhead irrigation to minimize leaf wetness. Apply protective copper soap.",
        "moderate_stage_advisory": "Apply protective and curative bio-fungicide (Bacillus subtilis) or Copper Oxychloride spray every 7-10 days.",
        "severe_stage_advisory": "Urgent intervention: Rotate contact and systemic fungicides such as Mancozeb (2.5g/L) or Chlorothalonil with Azoxystrobin. Check neighboring crops.",
        "organic_remedies": [
            "Cold-pressed Neem Oil spray (5ml/L of water with mild soap emulsifier)",
            "Bacillus subtilis biological formulation",
            "Copper hydroxide / Bordeaux mixture (1%)"
        ],
        "chemical_remedies": [
            "Mancozeb 75% WP @ 2.0 - 2.5 g/L water",
            "Chlorothalonil 75% WP @ 2.0 g/L water",
            "Azoxystrobin 23% SC @ 1.0 ml/L water"
        ],
        "cultural_practices": [
            "Practice 3-year crop rotation with non-solanaceous crops",
            "Use drip irrigation rather than overhead sprinklers",
            "Maintain optimal plant spacing (30-40 cm) for air circulation",
            "Ensure balanced nitrogen fertilization (excess N promotes vulnerability)"
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Water-soaked irregular black/brown lesions starting at leaf margins and tips, with white fungal sporulation underneath during humid weather.",
        "early_stage_advisory": "Critical threat: Inspect field immediately. Prune localized leaves, isolate section, and apply contact protectant fungicide.",
        "moderate_stage_advisory": "Apply systemic fungicide combination (Metalaxyl + Mancozeb). Cease all sprinkler irrigation.",
        "severe_stage_advisory": "Emergency epidemic risk: Spray systemic Cymoxanil / Dimethomorph combined with Mancozeb. If foliage is devastated, cut haulms to protect tubers.",
        "organic_remedies": [
            "Copper Oxychloride (50% WP) @ 3g/L as preventative barrier",
            "Trichoderma viride bio-fungicide soil and foliar drench"
        ],
        "chemical_remedies": [
            "Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ) @ 2.5 g/L",
            "Dimethomorph 50% WP @ 1.0 g/L",
            "Cymoxanil 8% + Mancozeb 64% WP @ 2.0 g/L"
        ],
        "cultural_practices": [
            "Plant certified disease-free certified seed tubers",
            "Hill up soil around stems to prevent zoospore tuber infection",
            "Destroy and bury all potato cull piles and volunteer plants",
            "Monitor regional weather alerts for humid, cool blight conditions"
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease_name": "Healthy Foliage",
        "pathogen": "None detected",
        "symptoms": "Vibrant emerald green leaves, intact cuticle, no chlorosis, necrosis, or irregular lesion formation.",
        "early_stage_advisory": "Foliage is healthy. Maintain regular monitoring and standard agronomic schedules.",
        "moderate_stage_advisory": "Not applicable.",
        "severe_stage_advisory": "Not applicable.",
        "organic_remedies": [
            "Prophylactic foliar spray of sea-weed extract or compost tea to boost natural immunity"
        ],
        "chemical_remedies": [
            "None required. Avoid unnecessary chemical applications to conserve beneficial microflora."
        ],
        "cultural_practices": [
            "Maintain consistent drip irrigation schedule",
            "Scout crop bi-weekly for early pest and disease onset"
        ]
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease_name": "Bacterial Spot",
        "pathogen": "Xanthomonas perforans / euvesicatoria (Bacteria)",
        "symptoms": "Small, dark brown to black circular lesions (1-3mm), often with greasy or water-soaked borders and conspicuous yellow chlorotic halos.",
        "early_stage_advisory": "Prune spotted leaves with sanitized shears. Avoid handling plants when wet. Apply fixed copper bactericide.",
        "moderate_stage_advisory": "Apply Copper Hydroxide tank-mixed with Mancozeb (creates synergistic bactericidal activity against copper-tolerant strains).",
        "severe_stage_advisory": "High defoliation danger: Apply Streptomycin Sulphate / Kasugamycin formulations where legally registered, combined with copper bactericide.",
        "organic_remedies": [
            "Copper Hydroxide 53.8% DF (Kocide) @ 1.5 - 2.0 g/L",
            "Bacillus amyloliquefaciens foliar spray",
            "Hydrogen dioxide / peroxyacetic acid sanitation spray"
        ],
        "chemical_remedies": [
            "Copper Oxychloride (50% WP) 2.5 g/L + Mancozeb 2.0 g/L",
            "Kasugamycin 3% SL @ 1.5 - 2.0 ml/L",
            "Streptomycin Sulfate 9% + Tetracycline 1% SP (Agrimycin)"
        ],
        "cultural_practices": [
            "Use certified pathogen-free hot-water treated tomato seeds",
            "Sterilize pruning shears with 70% isopropyl alcohol between rows",
            "Eliminate solanaceous weeds (nightshade) around the greenhouse/field",
            "Never work or prune in tomato fields while leaves are wet"
        ]
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "symptoms": "Dark brown circular spots with distinctive concentric target rings, starting on lower mature leaves and progressing upward.",
        "early_stage_advisory": "Remove lower 30cm of affected leaves. Apply mulching to prevent soil spores splashing onto lower leaves.",
        "moderate_stage_advisory": "Spray Chlorothalonil or Mancozeb every 7-10 days, ensuring full coverage of upper and lower leaf surfaces.",
        "severe_stage_advisory": "Apply systemic strobilurin or triazole fungicides (Azoxystrobin or Difenoconazole) to arrest upward canopy transmission.",
        "organic_remedies": [
            "Cold-pressed organic Neem oil (0.5% concentration)",
            "Liquid copper octanoate / Copper soap spray",
            "Potassium bicarbonate foliar spray (3g/L)"
        ],
        "chemical_remedies": [
            "Chlorothalonil 75% WP @ 2.0 g/L",
            "Difenoconazole 25% EC @ 0.5 - 0.8 ml/L",
            "Mancozeb 75% WP @ 2.5 g/L"
        ],
        "cultural_practices": [
            "Stake or cage tomato plants to elevate canopy off moist soil",
            "Lay down black plastic mulch or straw mulch around plant bases",
            "Implement 3-year rotation away from tomatoes, potatoes, eggplants",
            "Disinfect stakes and trellising wire at season end"
        ]
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Large, irregularly shaped greasy dark greenish-brown lesions on leaves and stems; rapid wilting and collapse during cool, damp weather.",
        "early_stage_advisory": "Emergency response: Immediately bag and remove infected plants. Cease overhead irrigation.",
        "moderate_stage_advisory": "Apply Metalaxyl + Mancozeb or Mandipropamid immediately to prevent canopy collapse.",
        "severe_stage_advisory": "Severe defoliation alert: Apply systemic Cymoxanil or Dimethomorph every 5-7 days under active blight conditions.",
        "organic_remedies": [
            "Bordeaux mixture (1%) preventive spray",
            "Copper sulfate pentahydrate bio-approved formulations"
        ],
        "chemical_remedies": [
            "Mandipropamid 23.4% SC (Revus) @ 0.8 ml/L",
            "Metalaxyl-M 4% + Mancozeb 64% WP @ 2.5 g/L",
            "Famoxadone + Cymoxanil (Equation Pro) @ 1.0 g/L"
        ],
        "cultural_practices": [
            "Select resistant cultivars (e.g., Defiant, Mountain Merit, Plum Regal)",
            "Provide maximum plant-to-plant ventilation in high tunnels",
            "Avoid planting downwind from potato fields"
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus (Transmitted by Silverleaf Whitefly - Bemisia tabaci)",
        "symptoms": "Severe stunting, erect/upright growth habit, cupping and curling of leaflets upward, with distinct interveinal and marginal yellowing.",
        "early_stage_advisory": "Rogue out infected plants immediately into sealed bags (cannot be cured once virus is systemic). Control whitefly vectors.",
        "moderate_stage_advisory": "Deploy yellow sticky cards (1 per 100 sq ft). Apply imidacloprid or acetamiprid systemic insecticide for whitefly suppression.",
        "severe_stage_advisory": "Field-wide vector extermination and destruction of all stunted symptomatic plants to protect adjacent blocks.",
        "organic_remedies": [
            "Yellow sticky traps for continuous whitefly monitoring and mass trapping",
            "Insecticidal soap or Horticultural mineral oil against whitefly nymphs",
            "Beauveria bassiana entomopathogenic fungal bio-insecticide"
        ],
        "chemical_remedies": [
            "Thiamethoxam 25% WG @ 0.3 g/L (vector control)",
            "Acetamiprid 20% SP @ 0.4 g/L",
            "Spiromesifen 22.9% SC @ 1.0 ml/L"
        ],
        "cultural_practices": [
            "Install 50-mesh insect-proof screening in greenhouse vents",
            "Use TYLCV-resistant hybrids (e.g., Tycoon, Charger, Invictus)",
            "Maintain a 2-month host-free period between tomato cropping cycles"
        ]
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease_name": "Healthy Foliage",
        "pathogen": "None detected",
        "symptoms": "Robust deep green foliage, crisp leaf margins, uniform color distribution, no viral curling or necrotic lesions.",
        "early_stage_advisory": "Foliage is healthy. Maintain regular fertigation and IPM inspection.",
        "moderate_stage_advisory": "Not applicable.",
        "severe_stage_advisory": "Not applicable.",
        "organic_remedies": [
            "Maintain beneficial companion planting (basil, marigolds) to repel whiteflies and nematodes"
        ],
        "chemical_remedies": [
            "None required."
        ],
        "cultural_practices": [
            "Continue uniform drip irrigation and calcium supplementation to prevent blossom end rot"
        ]
    },
    "Pepper__bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease_name": "Bacterial Leaf Spot",
        "pathogen": "Xanthomonas campestris pv. vesicatoria (Bacteria)",
        "symptoms": "Small, water-soaked, circular to irregular lesions on leaves; spots turn dark brown with pale centers and often cause severe defoliation.",
        "early_stage_advisory": "Prune spotted leaves with sanitized shears. Avoid overhead irrigation. Apply copper hydroxide bactericide.",
        "moderate_stage_advisory": "Apply Copper Hydroxide combined with Mancozeb to enhance copper bioavailability against resistant bacterial strains.",
        "severe_stage_advisory": "Apply Kasugamycin or oxytetracycline bactericide where permitted; heavily defoliated plants may need rogueing.",
        "organic_remedies": [
            "Liquid Copper Soap (Copper Octanoate) @ 15-20 ml/L",
            "Bacillus subtilis biological bactericide foliar treatment"
        ],
        "chemical_remedies": [
            "Copper Hydroxide 77% WP @ 2.0 g/L",
            "Copper Oxychloride 50% WP @ 2.5 g/L + Mancozeb 2.0 g/L",
            "Kasugamycin 3% SL @ 1.5 ml/L"
        ],
        "cultural_practices": [
            "Plant resistant pepper varieties (XVR series / races 1-10 resistant)",
            "Avoid handling plants during morning dew or following rain",
            "Sanitize all farm equipment, stakes, and pruning tools with 10% bleach"
        ]
    },
    "Pepper__bell___healthy": {
        "crop": "Bell Pepper",
        "disease_name": "Healthy Foliage",
        "pathogen": "None detected",
        "symptoms": "Uniform deep green glossy foliage, no chlorosis, leaf spots, or necrotic margins.",
        "early_stage_advisory": "Foliage is healthy. Continue routine agronomic management.",
        "moderate_stage_advisory": "Not applicable.",
        "severe_stage_advisory": "Not applicable.",
        "organic_remedies": [
            "Apply foliar micronutrient and compost tea spray to strengthen cell wall cuticles"
        ],
        "chemical_remedies": [
            "None required."
        ],
        "cultural_practices": [
            "Maintain soil pH between 6.0 and 6.8 with consistent moisture"
        ]
    }
}


def get_disease_remedy(predicted_class: str, severity_metrics: dict) -> dict:
    """
    Retrieves tailored treatment recommendations based on predicted class
    and quantitative severity percentage.
    """
    # Normalize class key
    info = REMEDIES_DATABASE.get(predicted_class)
    if info is None:
        # Fallback if class name slightly differs
        for key in REMEDIES_DATABASE:
            if key.lower() in predicted_class.lower() or predicted_class.lower() in key.lower():
                info = REMEDIES_DATABASE[key]
                break
                
    if info is None:
        # Generic fallback
        return {
            "crop": "Unknown Solanaceous Crop",
            "disease_name": predicted_class.replace("___", " - ").replace("_", " "),
            "pathogen": "Under diagnostic analysis",
            "symptoms": "Foliar abnormality detected.",
            "action_plan": "Isolate the plant, remove heavily symptomatic leaves, and consult local agriculture extension.",
            "organic_remedies": ["Apply organic neem oil spray (5ml/L)", "Bacillus subtilis foliar drench"],
            "chemical_remedies": ["Broad-spectrum protective copper fungicide (2.5 g/L)"],
            "cultural_practices": ["Avoid wetting foliage", "Improve spacing and ventilation"]
        }

    # Select severity-specific action plan
    sev_pct = severity_metrics.get("severity_percentage", 0.0)
    if sev_pct < 0.5 or "healthy" in predicted_class.lower():
        action_plan = "Plant is in optimal health. Maintain prophylactic scouting and standard irrigation."
    elif sev_pct <= 10.0:
        action_plan = info["early_stage_advisory"]
    elif sev_pct <= 25.0:
        action_plan = info["moderate_stage_advisory"]
    else:
        action_plan = info["severe_stage_advisory"]

    return {
        "crop": info["crop"],
        "disease_name": info["disease_name"],
        "pathogen": info["pathogen"],
        "symptoms": info["symptoms"],
        "action_plan": action_plan,
        "organic_remedies": info["organic_remedies"],
        "chemical_remedies": info["chemical_remedies"],
        "cultural_practices": info["cultural_practices"]
    }
