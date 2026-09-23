import os
import json
from flask import Flask, render_template, request, jsonify
from src.pipeline import ProverbPipeline
from src.evaluate import ProverbEvaluator

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)

# Initialize pipeline
pipeline = ProverbPipeline()
evaluator = ProverbEvaluator(pipeline)

# Cache benchmark results
benchmark_cache = None


def get_cached_benchmark():
    global benchmark_cache
    if benchmark_cache is not None:
        return benchmark_cache
    cache_path = os.path.join("results", "benchmark_results.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                benchmark_cache = json.load(f)
                return benchmark_cache
        except Exception:
            pass
    benchmark_cache = evaluator.run_full_benchmark()
    return benchmark_cache


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json(force=True, silent=True) or {}
    situation = data.get("situation", "").strip()
    model = data.get("model", "unified").strip()
    top_k = int(data.get("top_k", 5))

    if not situation:
        return jsonify({"error": "অনুগ্রহ করে একটি পরিস্থিতি বা ঘটনা লিখুন।"}), 400

    try:
        response = pipeline.retrieve(situation, model_name=model, top_k=top_k)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proverbs", methods=["GET"])
def get_proverbs():
    proverbs = pipeline.get_proverbs()
    query = request.args.get("q", "").strip().lower()
    if query:
        proverbs = [
            p for p in proverbs
            if query in str(p.get("proverb", "")).lower() 
            or query in str(p.get("meaning", "")).lower()
        ]
    return jsonify({"total": len(proverbs), "proverbs": proverbs})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "total_proverbs": len(pipeline.proverbs_df) if pipeline.proverbs_df is not None else 0,
        "total_situations": len(pipeline.train_df) if pipeline.train_df is not None else 0,
        "dev_situations": len(pipeline.dev_df) if pipeline.dev_df is not None else 0,
        "test_situations": len(pipeline.test_df) if pipeline.test_df is not None else 0,
    })


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    res = get_cached_benchmark()
    return jsonify(res)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return diverse sample situations for quick exploration."""
    samples = [
        {
            "label": "অতিরিক্ত লোভ ও ক্ষতি",
            "situation": "বেশি লাভের আশায় সে নিজের সঞ্চয় ও পৈতৃক জমি বিক্রি করে একটি অত্যন্ত ঝুঁকিপূর্ণ ও সন্দেহভাজন ব্যবসায় সব টাকা লাগাল। প্রতিষ্ঠানটি দেউলিয়া হওয়ায় সে সম্পূর্ণ নিঃস্ব হলো।",
            "expected": "অতি লোভে তাঁতি নষ্ট"
        },
        {
            "label": "নিজের অদক্ষতায় অজুহাত",
            "situation": "ড্রাইভিংয়ের কোনো প্রশিক্ষণ না নিয়ে গাড়ি চালাতে গিয়ে দেয়ালে লাগিয়ে চালক চিৎকার করে বলল গাড়ির স্টীয়ারিং আর ব্রেক নাকি ভালো ছিল না।",
            "expected": "নাচতে না জানলে উঠান বাঁকা"
        },
        {
            "label": "বিলম্বিত উপলব্ধি / আফসোস",
            "situation": "দোকানে বড় ধরনের চুরি হয়ে সব মালামাল ও ক্যাশ খোয়া যাওয়ার পর দোকানমালিক উপলব্ধি করলেন তার দোকানে আধুনিক তালা ও সিসিটিভি ক্যামেরা লাগানো খুব দরকার ছিল।",
            "expected": "চোর পালালে বুদ্ধি বাড়ে"
        },
        {
            "label": "অপ্রাপ্তিতে আগাম আনন্দ",
            "situation": "চাকরির মৌখিক পরীক্ষা দিয়ে এসে এখনো নিয়োগপত্র আসেনি, অথচ যুবকটি বাজারে গিয়ে বন্ধুদের দাওয়াত দিয়ে বিলাসবহুল পার্টি করার মিষ্টি কিনে ফেলল।",
            "expected": "গাছে কাঁঠাল গোঁফে তেল"
        },
        {
            "label": "উভয় পক্ষের দায়",
            "situation": "দুই সহকর্মীর তিক্ত বিবাদ ও ঝগড়ায় শুধু একজন দোষী ছিল না, অপরজনের উসকানিমূলক আচরণ ও আক্রমণাত্মক কথাও বিরোধ সৃষ্টির জন্য সমান দায়ী ছিল।",
            "expected": "এক হাতে তালি বাজে না"
        },
        {
            "label": "কঠোর পরিশ্রমের মহিমা",
            "situation": "টানা সাত বছর রাতদিন এক করে প্রতিকূলতার সাথে লড়াই ও ব্যক্তিগত আনন্দ বিসর্জন দিয়ে অবশেষে মেয়েটি আন্তর্জাতিক বিজ্ঞান অলিম্পিয়াডে স্বর্ণপদক জয় করল।",
            "expected": "কষ্ট না করলে কেষ্ট মেলে না / ধৈর্যের ফল মিষ্টি"
        },
        {
            "label": "অসৎ যোগসাজশ",
            "situation": "কালোবাজারি মাদক ব্যবসায়ী ও দুর্নীতিগ্রস্ত শুল্ক পরিদর্শক নিজেদের অবৈধ স্বার্থ বাঁচাতে গোপনে একে অপরকে নিরাপত্তা দেওয়ার সমঝোতা করল।",
            "expected": "চোরে চোরে মাসতুতো ভাই"
        }
    ]
    return jsonify(samples)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Probad Bangla Semantic Proverb Retrieval Web App on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
