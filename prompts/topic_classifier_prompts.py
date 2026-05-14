from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import os
from dotenv import load_dotenv
load_dotenv()

topic_data = [

# ================== Account Management ==================
{"query": "I’m excited to open my first savings account, how do I do it?", "topic": "Account Management", "confidence": "High", "sentiment_score": 5},
{"query": "I urgently need to convert my account into a salary account", "topic": "Account Management", "confidence": "High", "sentiment_score": 2},
{"query": "Why is updating my phone number so confusing?", "topic": "Account Management", "confidence": "Medium", "sentiment_score": 2},
{"query": "I’m frustrated, I can’t update my email in the bank app", "topic": "Account Management", "confidence": "High", "sentiment_score": 1},
{"query": "Can you help me change my account details please?", "topic": "Account Management", "confidence": "Medium", "sentiment_score": 3},
{"query": "I feel stuck with my account settings", "topic": "Account Management", "confidence": "Low", "sentiment_score": 2},
{"query": "How do I close my bank account safely?", "topic": "Account Management", "confidence": "High", "sentiment_score": 3},
{"query": "I’m confused about KYC process", "topic": "Account Management", "confidence": "Medium", "sentiment_score": 2},
{"query": "Account update issue is annoying me", "topic": "Account Management", "confidence": "Low", "sentiment_score": 2},
{"query": "Need help managing my account", "topic": "Account Management", "confidence": "Low", "sentiment_score": 3},
{"query": "Can I change my registered email and phone together?", "topic": "Account Management", "confidence": "High", "sentiment_score": 3},
{"query": "I want to update my profile details but it’s confusing", "topic": "Account Management", "confidence": "Medium", "sentiment_score": 2},

# ================== Payments & Transfers ==================
{"query": "I’m happy the UPI works but how do I send money?", "topic": "Payments & Transfers", "confidence": "High", "sentiment_score": 4},
{"query": "This is frustrating, my payment failed but money got deducted!", "topic": "Payments & Transfers", "confidence": "High", "sentiment_score": 1},
{"query": "Can you guide me on NEFT transfer?", "topic": "Payments & Transfers", "confidence": "High", "sentiment_score": 3},
{"query": "Why is my transaction stuck again?", "topic": "Payments & Transfers", "confidence": "Medium", "sentiment_score": 2},
{"query": "I’m worried, payment is not going through", "topic": "Payments & Transfers", "confidence": "Medium", "sentiment_score": 2},
{"query": "Transfer issue is really annoying", "topic": "Payments & Transfers", "confidence": "Medium", "sentiment_score": 1},
{"query": "I think something went wrong with my money transfer", "topic": "Payments & Transfers", "confidence": "Low", "sentiment_score": 2},
{"query": "Payment system seems broken today", "topic": "Payments & Transfers", "confidence": "Low", "sentiment_score": 1},
{"query": "How do I pay my electricity bill via bank?", "topic": "Payments & Transfers", "confidence": "High", "sentiment_score": 3},
{"query": "I feel confused about sending money", "topic": "Payments & Transfers", "confidence": "Low", "sentiment_score": 2},
{"query": "My transfer failed twice, what’s happening?", "topic": "Payments & Transfers", "confidence": "High", "sentiment_score": 1},
{"query": "How to send money to another bank account?", "topic": "Payments & Transfers", "confidence": "High", "sentiment_score": 3},

# ================== Security & Fraud ==================
{"query": "I’m scared my account got hacked!", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 1},
{"query": "Please help, someone made an unauthorized transaction!", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 1},
{"query": "I urgently want to block my debit card", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 2},
{"query": "This fraud call really worried me", "topic": "Security & Fraud", "confidence": "Medium", "sentiment_score": 2},
{"query": "I feel unsafe using the app now", "topic": "Security & Fraud", "confidence": "Medium", "sentiment_score": 2},
{"query": "There’s some suspicious activity in my account", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 2},
{"query": "I got an OTP request I didn’t initiate", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 1},
{"query": "Security issue maybe?", "topic": "Security & Fraud", "confidence": "Low", "sentiment_score": 3},
{"query": "Something doesn’t feel right with my account", "topic": "Security & Fraud", "confidence": "Low", "sentiment_score": 2},
{"query": "I think someone accessed my account", "topic": "Security & Fraud", "confidence": "Medium", "sentiment_score": 2},
{"query": "How can I secure my account better?", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 3},
{"query": "My card details might be compromised", "topic": "Security & Fraud", "confidence": "High", "sentiment_score": 1},

# ================== Balance & Statements ==================
{"query": "Can you show my account balance please?", "topic": "Balance & Statements", "confidence": "High", "sentiment_score": 3},
{"query": "I need my bank statement urgently", "topic": "Balance & Statements", "confidence": "High", "sentiment_score": 2},
{"query": "I’m confused where to check my transactions", "topic": "Balance & Statements", "confidence": "Medium", "sentiment_score": 2},
{"query": "Why can’t I see my balance?", "topic": "Balance & Statements", "confidence": "Medium", "sentiment_score": 2},
{"query": "Show me last 5 transactions", "topic": "Balance & Statements", "confidence": "High", "sentiment_score": 3},
{"query": "I feel lost trying to find my statement", "topic": "Balance & Statements", "confidence": "Low", "sentiment_score": 2},
{"query": "Balance details please", "topic": "Balance & Statements", "confidence": "Low", "sentiment_score": 3},
{"query": "How do I download statement PDF?", "topic": "Balance & Statements", "confidence": "High", "sentiment_score": 3},
{"query": "Transaction history is confusing me", "topic": "Balance & Statements", "confidence": "Medium", "sentiment_score": 2},
{"query": "Money details in account?", "topic": "Balance & Statements", "confidence": "Low", "sentiment_score": 3},
{"query": "Where can I check my recent transactions?", "topic": "Balance & Statements", "confidence": "High", "sentiment_score": 3},
{"query": "Statement download not working", "topic": "Balance & Statements", "confidence": "Medium", "sentiment_score": 2},

# ================== Loans & Credit ==================
{"query": "I’m interested in applying for a personal loan", "topic": "Loans & Credit", "confidence": "High", "sentiment_score": 4},
{"query": "Can you check my loan eligibility?", "topic": "Loans & Credit", "confidence": "High", "sentiment_score": 3},
{"query": "What will be EMI for my loan?", "topic": "Loans & Credit", "confidence": "High", "sentiment_score": 3},
{"query": "Loan status is making me anxious", "topic": "Loans & Credit", "confidence": "Medium", "sentiment_score": 2},
{"query": "I’m confused about loan process", "topic": "Loans & Credit", "confidence": "Medium", "sentiment_score": 2},
{"query": "Credit card bill seems too high", "topic": "Loans & Credit", "confidence": "High", "sentiment_score": 1},
{"query": "Loan issue, need help", "topic": "Loans & Credit", "confidence": "Low", "sentiment_score": 2},
{"query": "Can I get loan quickly?", "topic": "Loans & Credit", "confidence": "Medium", "sentiment_score": 3},
{"query": "I feel stressed about my EMI", "topic": "Loans & Credit", "confidence": "Medium", "sentiment_score": 1},
{"query": "Loan details please", "topic": "Loans & Credit", "confidence": "Low", "sentiment_score": 3},
{"query": "How long does loan approval take?", "topic": "Loans & Credit", "confidence": "High", "sentiment_score": 3},
{"query": "Why was my loan rejected?", "topic": "Loans & Credit", "confidence": "High", "sentiment_score": 2},

# ================== Cards & Services ==================
{"query": "How do I activate my debit card?", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 3},
{"query": "I lost my card, please block it immediately!", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 1},
{"query": "I’m frustrated my card is not working", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 1},
{"query": "How to generate ATM PIN?", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 3},
{"query": "Card issue is annoying me", "topic": "Cards & Services", "confidence": "Medium", "sentiment_score": 2},
{"query": "I feel worried about my card safety", "topic": "Cards & Services", "confidence": "Medium", "sentiment_score": 2},
{"query": "Card not working?", "topic": "Cards & Services", "confidence": "Low", "sentiment_score": 2},
{"query": "Need help with my credit card", "topic": "Cards & Services", "confidence": "Medium", "sentiment_score": 3},
{"query": "ATM declined my card, why?", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 2},
{"query": "Card problem", "topic": "Cards & Services", "confidence": "Low", "sentiment_score": 2},
{"query": "How to upgrade my credit card?", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 3},
{"query": "Why is my debit card blocked?", "topic": "Cards & Services", "confidence": "High", "sentiment_score": 2},

# ================== Financial Guidance ==================
{"query": "I want to save money better, any tips?", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 4},
{"query": "I feel stressed managing my expenses", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 1},
{"query": "Give me budgeting advice please", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 3},
{"query": "How can I improve my savings?", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 3},
{"query": "I’m confused about managing money", "topic": "Financial Guidance", "confidence": "Medium", "sentiment_score": 2},
{"query": "Any financial tips?", "topic": "Financial Guidance", "confidence": "Medium", "sentiment_score": 3},
{"query": "Help me manage finances", "topic": "Financial Guidance", "confidence": "Low", "sentiment_score": 2},
{"query": "I want to plan my budget", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 4},
{"query": "Money management is hard", "topic": "Financial Guidance", "confidence": "Medium", "sentiment_score": 2},
{"query": "Finance help needed", "topic": "Financial Guidance", "confidence": "Low", "sentiment_score": 2},
{"query": "How to reduce unnecessary expenses?", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 3},
{"query": "I want to save more each month", "topic": "Financial Guidance", "confidence": "High", "sentiment_score": 4},

# ================== Charges & Policies ==================
{"query": "Why was I charged for low balance?", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 1},
{"query": "Explain bank charges please", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 3},
{"query": "I’m confused about interest rates", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 2},
{"query": "Any penalty for minimum balance?", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 3},
{"query": "Charges seem unfair", "topic": "Charges & Policies", "confidence": "Medium", "sentiment_score": 1},
{"query": "What fees apply here?", "topic": "Charges & Policies", "confidence": "Medium", "sentiment_score": 3},
{"query": "Bank rules confusing me", "topic": "Charges & Policies", "confidence": "Low", "sentiment_score": 2},
{"query": "Tell me about account charges", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 3},
{"query": "Interest rate details please", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 3},
{"query": "Fees info?", "topic": "Charges & Policies", "confidence": "Low", "sentiment_score": 3},
{"query": "Why are extra charges applied?", "topic": "Charges & Policies", "confidence": "High", "sentiment_score": 2},
{"query": "I don’t understand bank policies", "topic": "Charges & Policies", "confidence": "Medium", "sentiment_score": 2},

# ================== Support & Feedback ==================
{"query": "I want to raise a complaint", "topic": "Support & Feedback", "confidence": "High", "sentiment_score": 2},
{"query": "I’m frustrated, customer support is not responding", "topic": "Support & Feedback", "confidence": "High", "sentiment_score": 1},
{"query": "How can I track my complaint?", "topic": "Support & Feedback", "confidence": "High", "sentiment_score": 3},
{"query": "I want to give feedback about the app", "topic": "Support & Feedback", "confidence": "High", "sentiment_score": 4},
{"query": "Support team is not helpful", "topic": "Support & Feedback", "confidence": "Medium", "sentiment_score": 2},
{"query": "Need help from customer service", "topic": "Support & Feedback", "confidence": "Medium", "sentiment_score": 2},
{"query": "I have an issue with the app", "topic": "Support & Feedback", "confidence": "Medium", "sentiment_score": 2},
{"query": "Help needed", "topic": "Support & Feedback", "confidence": "Low", "sentiment_score": 2},
{"query": "Complaint process confusing me", "topic": "Support & Feedback", "confidence": "Medium", "sentiment_score": 2},
{"query": "Facing issue", "topic": "Support & Feedback", "confidence": "Low", "sentiment_score": 2},
{"query": "How to contact customer care?", "topic": "Support & Feedback", "confidence": "High", "sentiment_score": 3},
{"query": "No response from support, very disappointed", "topic": "Support & Feedback", "confidence": "High", "sentiment_score": 1},

]


embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "banking-guardrails"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

vectorstore.add_texts(
    texts=[item["query"] for item in topic_data],
    metadatas=topic_data
)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=5,
    input_keys=["input"]
)

topic_example_prompt = PromptTemplate.from_template(
    "Query: {query}\nTopic: {topic}\nConfidence: {confidence}\nSentiment Score: {sentiment_score}"
)

topic_few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=topic_example_prompt,
    prefix=(
        "You are a topic classifier for a banking assistant.\n"
        "Classify the user query into the most relevant banking topic from the list below.\n"
        "Also assess your confidence level and assign a sentiment score based on the user's tone.\n\n"
        "Topics: Account Management, Payments & Transfers, Security & Fraud, Balance & Statements, "
        "Loans & Credit, Cards & Services, Financial Guidance, Charges & Policies, Support & Feedback\n\n"
        "Sentiment Score Guide:\n"
        "1 = Very negative (angry, distressed, urgent)\n"
        "2 = Negative (frustrated, worried, confused)\n"
        "3 = Neutral (informational, calm request)\n"
        "4 = Positive (eager, curious, happy)\n"
        "5 = Very positive (excited, highly satisfied)\n\n"
        "Relevant examples:"
    ),
    suffix=(
        "\nNow classify this:\n"
        "Query: {input}\n\n"
        "Respond with:\n"
        "Topic:\n"
        "Confidence:\n"
        "Sentiment Score:"
    ),
    input_variables=["input"]
)