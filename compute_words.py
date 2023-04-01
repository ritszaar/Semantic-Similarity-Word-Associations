import re
import pickle
import psycopg2

topIWK = 4
topWWK = 4

def connect():
    print("Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="word_associations",
            user="miranda",
            password="1234")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def get_base_words():
    base_words = set()
    for i in range(0, 50000, 1000):
        with open("./Base Associations/{}-{}.txt".format(i, i + 1000 - 1), "r") as f:
            for i in range(0, 11000, 11):
                fine_label = re.findall('"([^"]*)"', f.readline())[0]
                base_words.add(fine_label)
                for i in range(10):
                    if i < topIWK:
                        pred_label = re.findall('"([^"]*)"', f.readline())[0]
                        base_words.add(pred_label)
                    else:
                        f.readline()
    return sorted(base_words)

def get_all_words(conn):
    print("Getting all the words for the WAN...")
    all_words = set()
    base_words = get_base_words()
    for base_word in base_words:
        all_words.add(base_word)
        cue = base_word.split(" ")[-1].replace("'", "''")
        cursor = conn.cursor()
        cursor.execute("SELECT target, strength FROM usf_word_associations WHERE cue='{}'".format(cue))
        rows = cursor.fetchall()
        derived_words_count = min(len(rows) if rows is not None else 0, topWWK)
        for i in range(derived_words_count):
            all_words.add(rows[i][0])
        cursor.close()
    all_words = sorted(all_words)
    word2id, id2word = dict(), dict()
    with open("./Words/all_words.txt", "w") as f:
        print("{} words".format(len(all_words)), file=f)
        for i in range(len(all_words)):
            word2id[all_words[i]] = i
            id2word[i] = all_words[i]
            print('{} "{}"'.format(i, all_words[i]), file=f)
    print("Successfully obtained all the words for the WAN. {} words found.".format(len(all_words)))
    return all_words, word2id, id2word

print()
conn = connect()
if conn is not None:
    print("Successfully connected to PostgreSQL database.\n")
    all_words, word2id, id2word = get_all_words(conn)
    data = dict()
    data["all_words"] = all_words
    data["word2id"] = word2id
    data["id2word"] = id2word

    print('\nDumping data in "words.pickle"...')
    with open("words.pickle", "wb") as f:
        pickle.dump(data, file=f)
    print('Successfully dumped data in "words.pickle".')

    print("\nClosing the connection to PostgreSQL database...")
    conn.close()
    print("Successfully closed the connection to PostgreSQL database.\n")

