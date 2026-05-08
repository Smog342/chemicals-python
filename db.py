import psycopg

def changeTaskStatus(taskId, taskResult):
    conn = psycopg.connect(dbname="chemicals_db", user="postgres", password="root", host="localhost", port=5432)

    with conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE tasks SET result = '{taskResult}' WHERE id = {taskId}")