from app import create_app
from app.db import insert_cycle


def main() -> None:
    cycles = [
        {"start_date": "2025-07-07", "end_date": "2025-07-11"},
        {"start_date": "2025-08-03", "end_date": "2025-08-08"},
        {"start_date": "2025-09-04", "end_date": "2025-09-08"},
        {"start_date": "2025-10-08", "end_date": "2025-10-11"},
        {"start_date": "2025-11-11", "end_date": "2025-11-15"},
        # {"start_date": "2025-12-08", "end_date": "2025-12-13"},
    ]

    app = create_app()
    with app.app_context():
        for cycle in cycles:
            insert_cycle(cycle["start_date"], cycle["end_date"])

    print(f"Seeded {len(cycles)} cycles into the app database.")


if __name__ == "__main__":
    main()
