# Nepali Sentiment Analysis: Native Speaker Annotation Handbook
## Formal Annotation Guidelines for the NepSent-Engine Benchmark
**Version:** 2.0 (Ordinal Polarity Scheme)  
**Schema Version:** `2.0-ordinal`  
**Target Domains:** News & Politics, E-commerce Product Reviews, Social Media Discourse  
**Language:** Nepali (नेपाली भाषा) — Devanagari Script & Code-Mixed Expressions  

---

## 1. Introduction and Objectives

This handbook provides standardized, scientifically grounded guidelines for native Nepali speakers annotating sentiment polarity in text. 

The objective of **NepSent-Engine** is to build a high-reliability, reproducible sentiment benchmark across three domains with diverse linguistic styles:
1. **News & Politics (समाचार तथा राजनीति):** Formal, syntactic, often objective reporting mixed with political commentary.
2. **E-Commerce Reviews (अनलाइन किनमेल तथा समीक्षा):** Subjective, product- and service-oriented feedback, often polarized.
3. **Social Media (सामाजिक सञ्जाल):** Colloquial, slang-rich, code-mixed (Nepali-English), and sarcastic expressions.

To support ordinal evaluation metrics (such as linear-weighted Cohen's Kappa, $\kappa_w$), sentiment is annotated on a 3-point ordinal scale.

---

## 2. Ordinal Label Schema

| Label Code | Label Name | Meaning & Emotional Polarity |
|:---:|:---:|---|
| **`0`** | **Negative (नकारात्मक)** | Expresses dissatisfaction, anger, criticism, disappointment, sorrow, defectiveness, corruption, failure, or hostility. |
| **`1`** | **Neutral (तटस्थ)** | Objective factual reporting, balanced statements with no clear valence, unbiased inquiries, or non-evaluative information. |
| **`2`** | **Positive (सकारात्मक)** | Expresses satisfaction, admiration, praise, joy, appreciation, recommendation, success, optimism, or high quality. |

> [!IMPORTANT]
> **Ordinal Distance Principle:**
> Confusing **Negative (`0`)** with **Positive (`2`)** represents an extreme polarity reversal (error distance = 2). Confusing **Neutral (`1`)** with either Negative or Positive represents a borderline judgment (error distance = 1). Keep this distance in mind when distinguishing borderline cases.

---

## 3. General Decision Rules

### 3.1 Objective News vs. Subjective Neutral
A major challenge in low-resource sentiment annotation is news reporting. Factual events—even tragic or joyous ones—must be differentiated from subjective sentiment:

1. **Factual Reporting of Events $\rightarrow$ Neutral (`1`):**
   If a sentence reports a verifiable event objectively without authorial opinion, evaluation, or emotional adjectives, it is **Neutral (`1`)**, even if the event itself is negative or positive in real life.
   - *Example:* `"सडक दुर्घटनामा दुई जना घाइते भएका छन्।"` $\rightarrow$ **`1` (Neutral)** *(Objective crime/incident report; no evaluative author stance).*
   - *Example:* `"सरकारले आगामी आर्थिक वर्षको बजेट संसदमा प्रस्तुत गर्यो।"` $\rightarrow$ **`1` (Neutral)** *(Pure administrative fact).*
2. **Evaluative Commentary $\rightarrow$ Negative (`0`) or Positive (`2`):**
   If the sentence contains opinion adjectives, editorial blame, or emotional stance, annotate the polarity:
   - *Example:* `"सरकारको लापर्बाहीले गर्दा सडकको अवस्था अत्यन्तै दयनीय बनेको छ।"` $\rightarrow$ **`0` (Negative)** *(Evaluative: "लापर्बाहीले", "दयनीय").*
   - *Example:* `"नेपाल विद्युत प्राधिकरणको उत्कृष्ट व्यवस्थापनले लोडसेडिङ अन्त्य भएको छ।"` $\rightarrow$ **`2` (Positive)** *(Evaluative: "उत्कृष्ट व्यवस्थापन").*

---

### 3.2 Nepali Negation Morphology (नेपाली व्याकरणमा नकारात्मकता)
In Nepali, negation is rich, agglutinative, and appears in both inflected verb morphology and independent lexical particles. Annotators must carefully evaluate how negation inverts the core polarity.

#### Morphological & Synthetic Negation:
1. **Inflected Verb Suffixes (`-एन`, `-दैन`, `-दिन`, `-न्न`):**
   - `"यो सामान पटक्कै टिकेन।"` $\rightarrow$ **`0` (Negative)** *(Failure to last).*
   - `"उहाँको काम मलाई चित्त बुझेन।"` $\rightarrow$ **`0` (Negative)** *(Dissatisfaction).*
   - `"यो फोन छिटो तात्दैन।"` $\rightarrow$ **`2` (Positive)** *(Positive attribute: phone does not overheat).*
2. **Verbal Prefix `न-` (Affixed Negation):**
   - `"नराम्रो सेवा, कहिल्यै नलिनुहोला।"` $\rightarrow$ **`0` (Negative)** *(Affixed `न-` on adjective and imperative verb).*
   - `"कुनै समस्या नआई सामान समयमै आइपुग्यो।"` $\rightarrow$ **`2` (Positive)** *(Absence of problems = Positive).*
3. **Auxiliary Negation Particles (`छैन`, `होइन`, `हुँदैन`):**
   - `"क्यामेराको गुणस्तर राम्रो छैन।"` $\rightarrow$ **`0` (Negative)** *(Inverted positive).*
   - `"यो कुनै नराम्रो उत्पादन होइन, राम्रै छ।"` $\rightarrow$ **`2` (Positive)** *(Double negation / mitigating negative).*

---

### 3.3 Sarcasm and Irony (व्यङ्ग्य र कटाक्ष)
Nepali social media and political discourse make extensive use of sarcasm, where literal praise conceals bitter criticism.

> **The Sarcasm Law:** *Annotate the intended communicative meaning, NOT the literal surface words.*

- *Example:* `"वाह! हाम्रा नेताज्यूहरूको विकास त हेर्न लायक छ, एक झरीमै राजधानी पौडी पोखरी बन्यो!"`  
  - *Surface words:* "वाह!", "विकास", "हेर्न लायक" (literally positive).  
  - *Intended meaning:* Bitter mocking of infrastructure failure.  
  - *Annotation:* **`0` (Negative)** *(Tag `is_sarcastic: true` if using UI).*
- *Example:* `"१० दिनमा सामान डेलिभरी दिने दराजलाई धेरै धेरै धन्यवाद!"`  
  - *Surface words:* "धेरै धेरै धन्यवाद" (thank you).  
  - *Intended meaning:* Sarcastic frustration at 10-day delay.  
  - *Annotation:* **`0` (Negative)**.

---

### 3.4 Mixed Sentiment & Compound Clauses (मिश्रित भाव)
When a sentence contains both positive and negative clauses, apply the **Terminal Clause Priority Rule** and **Overall Weight Rule**:

1. **Adversative Conjunctions (`तर`, `तापनि`, `तर पनि`, `तैपनि`, `but`):**
   The clause following the adversative conjunction generally represents the author's final intended verdict.
   - *Example:* `"फोनको क्यामेरा धेरै राम्रो छ तर ब्याट्री एकदमै कमजोर लाग्यो।"`  
     $\rightarrow$ **`0` (Negative)** *(The concluding complaint overpowers the feature compliment).*
   - *Example:* `"मूल्य अलि महँगो भए पनि गुणस्तर साँच्चै उत्कृष्ट छ।"`  
     $\rightarrow$ **`2` (Positive)** *(Despite the price, concluding endorsement is strongly positive).*
2. **Equally Balanced Sentences $\rightarrow$ Neutral (`1`):**
   If the sentence balances positive and negative aspects with equal weight and no clear lean, label it **Neutral (`1`)**.
   - *Example:* `"यसमा केही राम्रा पक्ष छन् र केही कमजोरी पनि देखिन्छन्।"` $\rightarrow$ **`1` (Neutral)**.

---

### 3.5 Code-Mixed Nepali-English (मिश्रित भाषा)
Nepali speakers frequently mix English terms and Romanized phrases in reviews and social posts. 

- Annotate according to the overall sentiment regardless of language mixing:
  - `"Camera quality ekdam best cha, worth the price!"` $\rightarrow$ **`2` (Positive)**.
  - `"Size fit vayena, delivery pani ekdam delay vayo, totally waste of money."` $\rightarrow$ **`0` (Negative)**.
  - `"Order track garne option kata cha hola?"` $\rightarrow$ **`1` (Neutral)** *(Informational inquiry).*

---

## 4. Comprehensive Worked Examples by Domain

### 4.1 News & Politics (समाचार र राजनीति)

| Sentence | Correct Label | Explanation |
|---|:---:|---|
| `"प्रधानमन्त्रीले आज बिहान १० बजे मन्त्रिपरिषद्को बैठक बोलाएका छन्।"` | **1 (Neutral)** | Factual schedule; zero subjective evaluation. |
| `"भ्रष्टाचारमा संलग्न जोसुकै भए पनि कडा कारबाही गरिनेछ: गृहमन्त्री"` | **1 (Neutral)** | Attribution of formal quote/policy statement. |
| `"सांसदहरूको गैरजिम्मेवार व्यवहारले गर्दा आजको संसद बैठक स्थगित भयो।"` | **0 (Negative)** | Explicit evaluative judgment ("गैरजिम्मेवार व्यवहार"). |
| `"नेपाली क्रिकेट टोलीले विश्वकपमा ऐतिहासिक जित हासिल गर्दै देशको गौरव बढायो।"` | **2 (Positive)** | Celebratory language ("ऐतिहासिक जित", "गौरव बढायो"). |

### 4.2 E-Commerce Reviews (उत्पादन समीक्षा)

| Sentence | Correct Label | Explanation |
|---|:---:|---|
| `"कपडाको रङ फोटोमा देखिए जस्तै छ र कपडा पनि नरम छ, म सन्तुष्ट छु।" | **2 (Positive)** | Explicit satisfaction and praise. |
| `"डेलिभरी ढिलो भयो तर सामान राम्रो अवस्थामा पाइयो।" | **2 (Positive)** | Terminal clause confirms satisfaction with product. |
| `"फोटोमा एउटा देखायो, पठाउँदा अर्कै नक्कली सामान पठायो, ठग विक्रेता!" | **0 (Negative)** | Accusation of fraud and counterfeit goods. |
| `"प्याकेट भित्र वारेन्टी कार्ड समावेश गरिएको छ कि छैन?" | **1 (Neutral)** | Inquisitive question without sentiment. |

### 4.3 Social Media Discourse (सामाजिक सञ्जाल)

| Sentence | Correct Label | Explanation |
|---|:---:|---|
| `"आजको मौसम असाध्यै रमाइलो छ, चिया पिउँदै आनन्द लिनुपर्छ!"` | **2 (Positive)** | Expression of joy and relaxation. |
| `"जताततै धुलो र खाल्डाखुल्डी मात्र छ, कहिले बन्छ यो सडक?"` | **0 (Negative)** | Frustration, grievance, rhetorical complaint. |
| `"आज त्रिभुवन अन्तर्राष्ट्रिय विमानस्थलबाट ५ वटा अन्तर्राष्ट्रिय उडान भए।"` | **1 (Neutral)** | Unbiased statistical fact. |
| `"वाह नेताज्यू! महँगी बढाएर जनतालाई राम्रो उपहार दिनुभयो!"` | **0 (Negative)** | Sarcastic criticism of price inflation. |

---

## 5. Annotator Quality Assurance & Agreement Process

1. **Independent Blinding:** Annotators must label sheets independently without consulting each other or external model outputs.
2. **Confidence Ratings:** Each sample should be marked with confidence:
   - `HIGH`: Unambiguous sentiment.
   - `MEDIUM`: Slight ambiguity in phrasing or tone.
   - `LOW`: Highly mixed, culturally ambiguous, or obscure context.
3. **Disagreement Adjudication:** Sentences where Annotator A and Annotator B disagree will be flagged automatically. Disagreements will be reviewed in a consensus session or resolved by a third senior adjudicator.
4. **Target Agreement:** The campaign aims for Linear-Weighted Kappa $\kappa_w \ge 0.65$ on double-annotated validation sets.

