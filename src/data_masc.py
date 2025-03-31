import json
import sqlite3
import re
import gzip
import datetime
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import wordnet as wn


def get_omsti(cursor, counter):
    with gzip.open("data/one_million.json.gz", "r") as fin:
        data = json.load(fin)
        print("Getting one_million data.")
        for instance in tqdm(data):
            key = instance["sense_keys"][0]
            lemma = instance["sense_keys"][0].split("%")[0]
            if counter[key] > 2:
                continue
            counter[key] += 1
            instance_data = (
                key,
                lemma,
                instance["context"].replace("\n", " "),
                "one_million"
            )
            if lemma not in instance["context"]:
                continue
            cursor.execute("""
            INSERT INTO INSTANCES
            (SENSE_KEY, LEMMA, USAGE, ORIGIN)
            VALUES (?, ?, ?, ?)
            """, instance_data)


def synset_to_sense_key(synset):
    """
    Convert a WordNet synset to its corresponding sense keys.

    Args:
        synset (wn.Synset): A WordNet synset.

    Returns:
        list: A list of sense keys for the given synset.
    """
    lem_s = synset.split(".")[0]
    synset = wn.synset(synset)
    sense_keys = [
            lemma.key() for lemma in synset.lemmas() if lemma.name() == lem_s]
    return sense_keys


def get_semcor(cursor, counter):
    with gzip.open("data/semcor_en.jsonl.gz", "r") as fin:
        data = json.load(fin)
        print("Getting semcor data.")
        for item in tqdm(data):
            syn = [x for x in item["synsets"] if x.startswith(item["lemma"])]
            if syn:
                key = synset_to_sense_key(syn[0])
            if syn and key:
                key = key[0]
                if counter[key] > 2:
                    continue
                counter[key] += 1
                instance_data = (
                    key,
                    item["lemma"],
                    item["text"].strip(),
                    "semcor"
                )
                cursor.execute("""
                INSERT INTO INSTANCES
                (SENSE_KEY, LEMMA, USAGE, ORIGIN)
                VALUES (?, ?, ?, ?)
                """, instance_data)


def get_fews(cursor, counter):
    with gzip.open("data/fews.jsonl.gz", "r") as fin:
        data = json.load(fin)
        print("Getting fews data.")
        for item in tqdm(data):
            key = item["key"]
            if counter[key] > 2:
                continue
            counter[key] += 1
            instance_data = (
                key,
                item["lemma"],
                re.sub(r"</?WSD>", "", item["usage"]),
                "fews"
            )
            cursor.execute("""
            INSERT INTO INSTANCES
            (SENSE_KEY, LEMMA, USAGE, ORIGIN)
            VALUES (?, ?, ?, ?)
            """, instance_data)


def get_masc(cursor, counter):
    with gzip.open("data/masc.json.gz", "r") as fin:
        data = json.load(fin)
        print("Getting masc data.")
        for item in tqdm(data):
            key = item["sense_key"]
            if counter[key] > 2:
                continue
            counter[key] += 1
            instance_data = (
                key,
                item["sense_key"].split("%")[0].split(";")[0],
                item["usage"],
                "masc"
            )
            cursor.execute("""
            INSERT INTO INSTANCES
            (SENSE_KEY, LEMMA, USAGE, ORIGIN)
            VALUES (?, ?, ?, ?)
            """, instance_data)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"Start Time: {start_time}")

    conn = sqlite3.connect("usage.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS INSTANCES")
    cursor.execute("""
        CREATE TABLE INSTANCES (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            SENSE_KEY TEXT,
            LEMMA TEXT,
            USAGE TEXT,
            ORIGIN TEXT
        )
    """)
    counter = defaultdict(int)
    get_semcor(cursor, counter)
    print("In db data:",
          cursor.execute("SELECT COUNT(*) FROM INSTANCES").fetchone())
    get_masc(cursor, counter)
    print("In db data:",
          cursor.execute("SELECT COUNT(*) FROM INSTANCES").fetchone())
    # get_omsti(cursor, counter)
    # print("In db data:",
    #       cursor.execute("SELECT COUNT(*) FROM INSTANCES").fetchone())

    # Delete instances that have a single registered sense key.
    print("Remove entries with unique sense.")
    cursor.execute("""
        DELETE FROM INSTANCES
        WHERE SENSE_KEY IN (
            SELECT SENSE_KEY
            FROM INSTANCES
            GROUP BY SENSE_KEY
            HAVING COUNT(*) = 1
        )
    """)
    print("In db data:",
          cursor.execute("SELECT COUNT(*) FROM INSTANCES").fetchone())

    # Create index for fast join
    cursor.execute(""" CREATE INDEX IDX_LEMMA ON INSTANCES(LEMMA) """)

    print("Create table with pair of usages.")
    cursor.execute("DROP TABLE IF EXISTS PAIRS")
    cursor.execute("""
        CREATE TABLE PAIRS AS
        SELECT
            t1.LEMMA AS LEMMA,
            t1.SENSE_KEY AS SENSE_KEY_1,
            t2.SENSE_KEY AS SENSE_KEY_2,
            t1.USAGE AS USAGE_1,
            t2.USAGE AS USAGE_2
        FROM INSTANCES t1
        JOIN INSTANCES t2
        ON t1.LEMMA = t2.LEMMA
        WHERE t1.ID < t2.ID
    """)
    print("In db data:",
          cursor.execute("SELECT COUNT(*) FROM PAIRS").fetchone())

    conn.commit()
    conn.close()

    end_time = datetime.datetime.now()
    print(f"End Time: {end_time}")
