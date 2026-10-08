from app.services.retrieval_service import RetrievalService


RFP_ID = "cef53ab4-84a8-407d-aef7-176f7926b306"


"""
Evaluation data for RFP_DPR.pdf (Bhopal-Kota 4-laning DPR, NHAI) - Hit@k.

Page numbers = physical PDF page numbers, 1-based (page 1 = cover page).
Printed page labels inside the PDF are different, so do NOT use those.
If your Qdrant payload stores 0-based page numbers (e.g. PyMuPDF page.number),
use: expected_pages = [p - 1 for p in item["expected_pages"]]
"""

EVALUATION_DATA = [
    {
        # TIS Appendix-1 cl. 1.1-1.2: entity type, registrations, not insolvent/blacklisted, cooling-off, conflict of interest
        "question": "What are the eligibility requirements for a consultant to participate in this RFP?",
        "expected_pages": [14, 15],
    },
    {
        # TIS cl. 1.7: DPR aggregate length = package length; one project >= 40% (or feasibility >= 60%); last 10 years
        "question": "What is the minimum DPR experience required from a firm for this package?",
        "expected_pages": [17],
    },
    {
        # TIS cl. 1.7: avg annual turnover >= Rs 10 Cr (last 5 yrs); cl. 1.7 note 4: audited balance sheet / banking reference
        "question": "What are the financial eligibility criteria and which financial documents must be submitted?",
        "expected_pages": [17, 18],
    },
    {
        # TIS cl. 1.5.1: only 1 JV partner, international firm, 40% / lead 60%; Section VII A2 note 3 repeats it
        "question": "How many JV partners are allowed and what conditions must the JV partner meet?",
        "expected_pages": [16, 284],
    },
    {
        # TIS cl. 1.8 + Form T-11: R = CL x TF - RP; TF = 1.00 / 1.25 / 1.50 by average turnover
        "question": "How is the Residual DPR Bid Capacity (R) calculated and what turnover factors are applied?",
        "expected_pages": [18, 19, 321, 322],
    },
    {
        # TIS cl. 1.4 + ITC 3.1(4): max 10% of assignment cost, only specialized survey & investigation, prior approval
        "question": "What is the limit and condition for sub-contracting in this assignment?",
        "expected_pages": [15, 25],
    },
    {
        # TIS cl. 7.0 + ITC 11.4.1(11) + SCC 5.8: min 5%, higher of 5% of contract price and quoted PBG, JV members separately
        "question": "What is the minimum Performance Bank Guarantee required and how should JV members furnish it?",
        "expected_pages": [13, 40, 99],
    },
    {
        # ITC 11.4.1(11): QBS weights 30:30:40 (technical : DPR rating : PBG quote), formulas St, Sf and S
        "question": "How is the final combined score calculated for selecting the H-1 bidder (weightage of technical score, DPR rating and PBG quote)?",
        "expected_pages": [40, 41],
    },
    {
        # Section VII: A1 = 5, A2 = 30, B1 = 2.5, B2 = 2.5, C (key personnel) = 60, total 100
        "question": "What is the maximum marks distribution for the technical proposal evaluation?",
        "expected_pages": [282],
    },
    {
        # Section VI-A Enclosure-A (10 key experts, man-months) + BOQ Form-III (positions, SM, rates)
        "question": "Which key experts are required and what is the man-month input of each?",
        "expected_pages": [280, 355],
    },
    {
        # GCC 10.5.1(10): stage-wise payment % of contract price (First / Second / Third stage + PBG release)
        "question": "What is the stage-wise payment schedule for the consultant as a percentage of contract price?",
        "expected_pages": [78, 79],
    },
    {
        # GCC 14.3.2: 0.05% of contract price per day, max 5% of contract value
        "question": "What penalty is imposed for delay in completion of services?",
        "expected_pages": [71, 96],
    },
    {
        # TOR cl. 9.2: eight stages (Inception, Feasibility, LA & Clearances I, DPR, Technical Schedules, LA II, LA III, LA IV)
        "question": "Into how many stages is the DPR project preparation divided and what is the deliverable of each stage?",
        "expected_pages": [151],
    },
    {
        # TOR cl. 4.9.2(4): 7 days continuous, direction-wise
        "question": "For how many days must the classified traffic volume count survey be carried out?",
        "expected_pages": [113],
    },
    {
        # Appendix C Form-II: Rs 9,86,44,500 (net of GST), Rs 11,64,00,510 (with GST), per km cost, 340 km
        "question": "What is the estimated consultancy cost and per km DPR cost for this assignment?",
        "expected_pages": [353],
    },
]


def test_retrieval_evaluation():
    service = RetrievalService()

    hit_at_1 = 0
    hit_at_2 = 0
    hit_at_3 = 0
    hit_at_4 = 0
    hit_at_5 = 0

    for item in EVALUATION_DATA:
        question = item["question"]
        expected_pages = item["expected_pages"]

        results = service.search(
            query=question,
            rfp_id=RFP_ID,
            top_k=5,
        )

        retrieved_pages = [result["page_number"] for result in results]

        hit1 = any(page in expected_pages for page in retrieved_pages[:1])

        hit2 = any(page in expected_pages for page in retrieved_pages[:2])

        hit3 = any(page in expected_pages for page in retrieved_pages[:3])

        hit4 = any(page in expected_pages for page in retrieved_pages[:4])

        hit5 = any(page in expected_pages for page in retrieved_pages[:5])

        hit_at_1 += hit1
        hit_at_2 += hit2
        hit_at_3 += hit3
        hit_at_4 += hit4
        hit_at_5 += hit5

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print(f"EXPECTED PAGES: {expected_pages}")
        print(f"RETRIEVED PAGES: {retrieved_pages}")

        print(f"Hit@1: {'YES' if hit1 else 'NO'}")
        print(f"Hit@2: {'YES' if hit2 else 'NO'}")
        print(f"Hit@3: {'YES' if hit3 else 'NO'}")
        print(f"Hit@4: {'YES' if hit4 else 'NO'}")
        print(f"Hit@5: {'YES' if hit5 else 'NO'}")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"score={result['score']:.4f} "
                f"page={result['page_number']} "
                f"chunk={result['chunk_id']}"
            )

    total = len(EVALUATION_DATA)

    hit_at_1_score = hit_at_1 / total
    hit_at_2_score = hit_at_2 / total
    hit_at_3_score = hit_at_3 / total
    hit_at_4_score = hit_at_4 / total
    hit_at_5_score = hit_at_5 / total

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1_score:.2%}")
    print(f"Hit@2: {hit_at_2}/{total} = {hit_at_2_score:.2%}")
    print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3_score:.2%}")
    print(f"Hit@4: {hit_at_4}/{total} = {hit_at_4_score:.2%}")
    print(f"Hit@5: {hit_at_5}/{total} = {hit_at_5_score:.2%}")

    assert hit_at_5_score >= 0.70
