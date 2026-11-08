# NepSent-Engine: Ethics Statement & Responsible Research Practice

---

## 1. Ethical Principles

NepSent-Engine is developed under the core ethical commitments of:
1. **Academic Transparency:** Explicitly disavowing simulated or fabricated benchmarks.
2. **Responsible Data Collection:** Protecting content creator privacy and adhering to intellectual property boundaries.
3. **Linguistic Equity:** Enhancing natural language technology for the low-resource Nepali language without extractive data practices.

---

## 2. Responsible Web Collection & Privacy

### 2.1 Scraping Etiquette & Rate Limiting
All automated data collection utilities implemented in NepSent-Engine adhere to ethical scraping guidelines:
- **Politeness Headers:** Explicit User-Agent identification and contact metadata.
- **Rate-Limiting:** Exponential backoff with inter-request delays ($\ge 1.0\text{--}2.0\text{ s}$) to prevent denial-of-service or server strain on local Nepali news publishers and online stores.
- **Directives:** Respecting `robots.txt` disallow paths.

### 2.2 Personally Identifiable Information (PII) Redaction
To protect the privacy of private individuals:
- Public social media handles (`@username`) are scrubbed or replaced with generic tokens (`@user`).
- Telephone numbers, email addresses, order tracking IDs, and physical addresses are automatically filtered out during ingestion.

---

## 3. Fair Dealing & Copyright Compliance

Text samples collected for research purposes are governed by the doctrine of **Fair Dealing / Fair Use for non-commercial academic research and text/data mining (TDM)**.
- NepSent-Engine does NOT redistribute full proprietary news archives or commercial databases.
- The repository distributes code, processing tools, and annotation guidelines under the MIT License; text corpora remain the intellectual property of original authors and platforms.

---

## 4. Dual-Use & Downstream Societal Impact

### 4.1 Potential Harms
Automated sentiment classification can potentially be misused for:
- Mass political surveillance or censorship of dissident Nepali discourse.
- Unfair manipulation of merchant ratings on local e-commerce platforms.

### 4.2 Mitigation Strategies
- NepSent-Engine produces calibrated uncertainty metrics, preventing black-box overconfidence.
- We document extensive failure modes in [`LIMITATIONS.md`](LIMITATIONS.md), warning against using sentiment scores as sole decision criteria in high-stakes socio-political contexts.

