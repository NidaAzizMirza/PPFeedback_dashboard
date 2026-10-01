import sqlite3

conn = sqlite3.connect("/Users/nidaaziz/PycharmProjects/autoanalysisUserFeedback/data/metrics.db")

conn.execute("""
    DELETE FROM topic_monthly
    WHERE topic IN ('Overall Positive Experience', 'Negative experience')
""")

conn.commit()
conn.close()

print("Rows deleted successfully.")
