import os
from langchain_openai import ChatOpenAI
from graph import AgentState, build_graph
from datetime import date, timedelta

START_DATE = date(2026, 2, 2)   # 02 Feb 2026 IST
END_DATE   = date(2026, 3, 9)   # 09 Mar 2026 IST

PG_CONFIG = {
    "host":     os.environ.get("PG_HOST",     "localhost"),
    "port":     int(os.environ.get("PG_PORT", "5432")),
    "dbname":   os.environ.get("PG_DBNAME",   "glue-dispenser-db"),
    "user":     os.environ.get("PG_USER",     "postgres"),
    "password": os.environ.get("PG_PASSWORD", "postgres"),
}

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.2.133:8000/v1")
VLLM_MODEL    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen3-14B-AWQ")
SUMMARY_TABLE = os.environ.get("SUMMARY_TABLE", "agent_summaries")

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model=VLLM_MODEL,
    openai_api_base=VLLM_BASE_URL,
    openai_api_key="EMPTY",
    temperature=0.2,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# ─────────────────────────────────────────────
# RUN — one iteration per day
# ─────────────────────────────────────────────

def run():

    app = build_graph(
        pg_config=PG_CONFIG,
        llm=llm,
        summary_table=SUMMARY_TABLE,
        vllm_base_url=VLLM_BASE_URL,
        vllm_model=VLLM_MODEL,
    )

    current = START_DATE
    total   = (END_DATE - START_DATE).days
    done    = 0

    print(f"Processing {total} days from {START_DATE} to {END_DATE - timedelta(days=1)}\n")

    while current < END_DATE:
        next_day = current + timedelta(days=1)

        # IST timestamps passed into every query function
        start_ts = f"{current} 00:00:00+05:30"
        end_ts   = f"{next_day} 00:00:00+05:30"

        app.invoke({
            "run_date": str(current),
            "start":    start_ts,
            "end":      end_ts,
            "summary":  "",
        })

        done   += 1
        current = next_day
        print(f"  [{done}/{total}] {current - timedelta(days=1)} done\n")

    print(f"\nAll {total} daily summaries written to `{SUMMARY_TABLE}`.")


if __name__ == "__main__":
    run()