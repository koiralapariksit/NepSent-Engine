"""
Generate synthetic Nepali test dataset for pipeline demonstration.
This creates realistic sample data to verify the full scraping → processing → annotation pipeline.
"""

import json
from pathlib import Path
from datetime import datetime
import random

# Synthetic Nepali text samples (real sentences)
NEPALI_NEWS_SAMPLES = [
    "नेपालको अर्थनीति यस वर्ष बढदारी गरिरहेको छ।",
    "काठमाडौंमा आज सबेरे मूसलधार वर्षा भएको छ।",
    "सरकारले नयाँ शिक्षा नीति घोषणा गरेको छ।",
    "भारत र नेपालबीच द्विपक्षीय वार्ता सफल भएको छ।",
    "गत महिना विदेशी पर्यटकको आगमन बढेको छ।",
    "राष्ट्रपति राष्ट्रिय दिवसको अवसरमा सम्बोधन गर्नुहुनेछ।",
    "नेपाली क्रिकेट टीम अन्तर्राष्ट्रिय खेलमा जयी भएको छ।",
    "काठमाडौंको ट्राफिक समस्या गम्भीर बनिरहेको छ।",
    "नेपালमा सामाजिक माध्यम प्रयोगकर्ता संख्या बढ्दै छ।",
    "सोल्टी खोला पुल निर्माण कार्य शुरु भएको छ।",
]

NEPALI_REVIEWS_SAMPLES = [
    "यो उत्पाद धेरै राम्रो छ र मलाई मनपर्यो।",
    "मूल्य सामान्य छ तर गुणस्तर उत्कृष्ट छ।",
    "डेलिभरी समय छिटो र पयकिङ गरिएको छ।",
    "यो खरीद गर्न बहुत पछतای हुँ, खराब उत्पाद।",
    "नयाँ कपडा पस्ट याद गर्न सहजै धुइए पनि हराए।",
    "मोबाइल फोन बहुत राम्रो छ, सिफारस गर्दछु।",
    "जुत्ता आरामदायक र टिकाऊ छ।",
    "यो किताब पढ्न बहुत रोचक छ।",
    "खाद्य पदार्थ ताज र स्वादिलो।",
    "सेवामूलक अनुभव बहुत राम्रो रहेको छ।",
]

NEPALI_SOCIAL_SAMPLES = [
    "आज मौसम कस्तो राम्रो छ! बाहिर जान मन लागे।",
    "नेपालको मुक्तिका लागि देशभक्तहरू कुरबानी दिएका छन्।",
    "धेरै खुसी छु नयाँ नोकरी पाएर।",
    "अनौठो कहानी सुनें: गाउँमा एकै रातमा बिजुली आयो।",
    "सरकारको नीति साधारण जनताविरुद्ध छ।",
    "पढाइ गर्दा यो विषय सबैलाई कठिन लाग्छ।",
    "काठमाडौंमा रहेको नेपाल संग्रहालय दर्शन गरें।",
    "राष्ट्रको प्रगतिका लागि सबैको प्रयास आवश्यक छ।",
    "परिवारसँग छुट्टीमा हिमाल यात्रा गर्न गएका छौँ।",
    "यो गान सुनेर मेरो मन भर गयो।",
]

def generate_synthetic_corpus():
    """Generate and save synthetic raw corpus JSONL."""
    
    data_dir = Path("d:/ds mdoel/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_corpus_file = data_dir / "01_raw_corpus.jsonl"
    
    print("[*] Generating synthetic Nepali corpus...")
    
    records = []
    
    # Generate news records
    for i, text in enumerate(NEPALI_NEWS_SAMPLES):
        for platform in ["ratopati", "setopati", "onlinekhabar"]:
            records.append({
                "domain": "news_politics",
                "platform": platform,
                "url": f"https://{platform}.com/news/{i}",
                "headline": text[:50] + "...",
                "content": text + " " + text,  # Duplicate for realistic HTML
                "scraped_at": datetime.now().isoformat()
            })
    
    # Generate e-commerce records
    for i, text in enumerate(NEPALI_REVIEWS_SAMPLES):
        for j in range(2):  # Multiple reviews per product (simulated)
            records.append({
                "domain": "ecommerce_reviews",
                "platform": "daraz",
                "product_name": f"Product {i}",
                "review_text": text,
                "url": f"https://daraz.com.np/product/{i}",
                "scraped_at": datetime.now().isoformat()
            })
    
    # Generate social media records
    for i, text in enumerate(NEPALI_SOCIAL_SAMPLES):
        records.append({
            "domain": "social_media",
            "platform": "twitter",
            "tweet_id": str(1000000 + i),
            "text": text,
            "created_at": datetime.now().isoformat(),
            "public_metrics": {
                "like_count": random.randint(1, 50),
                "retweet_count": random.randint(0, 20)
            },
            "scraped_at": datetime.now().isoformat()
        })
    
    # Write JSONL
    with open(raw_corpus_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ Generated {len(records)} synthetic records")
    print(f"  - News & Politics: {sum(1 for r in records if r['domain'] == 'news_politics')}")
    print(f"  - E-commerce: {sum(1 for r in records if r['domain'] == 'ecommerce_reviews')}")
    print(f"  - Social Media: {sum(1 for r in records if r['domain'] == 'social_media')}")
    print(f"✓ Saved to: {raw_corpus_file}\n")
    
    return raw_corpus_file

if __name__ == "__main__":
    generate_synthetic_corpus()
