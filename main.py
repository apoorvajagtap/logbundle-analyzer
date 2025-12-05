# main.py

from runtime.query_engine import retrieve_kb, retrieve_sb_logs
from runtime.llm_reasoner import reason
import sys

def ask_question(question: str):
    print("\n[+] Retrieving KB matches...")
    kb_hits = retrieve_kb(question)

    print("[+] Retrieving SupportBundle log matches...")
    sb_hits = retrieve_sb_logs(question)

    print("[+] Calling LLM for final reasoning...\n")
    result = reason(question, kb_hits, sb_hits)

    print("==== Final Response ====\n")
    print(result["answer"])
    print("\n========================\n")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter your question: ")
    else:
        question = " ".join(sys.argv[1:])

    ask_question(question)
