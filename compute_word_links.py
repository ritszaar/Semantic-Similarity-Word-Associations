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

def load_data():
    print('Loading data from "word.pickle"...')
    f = open("words.pickle", "rb")
    data = pickle.load(f)
    f.close()
    print('Successfully loaded data from "word.pickle".')
    return data["all_words"], data["word2id"], data["id2word"]

def get_word_links(conn, all_words, word2id):
    print("\nGetting the word links...\n")
    word_links = [[-1] * len(all_words) for i in range(len(all_words))] 
    count = 0
    total_count = len(all_words) ** 2   
    for i in range(len(all_words)):
        for j in range(len(all_words)):
            cue = all_words[i].split(" ")[-1].replace("'", "''")
            target = all_words[j].split(" ")[-1].replace("'", "''")
            cursor = conn.cursor()
            cursor.execute("SELECT strength FROM usf_word_associations WHERE cue='{}' AND target='{}'".format(cue, target))
            row = cursor.fetchone()
            if row is not None:
                word_links[word2id[all_words[i]]][word2id[all_words[j]]] = float(row[0])
            cursor.close() 
            count = count + 1
            if count % 1000 == 0:
                print("Computed word strengths ({}/{}).".format(count, total_count))
    print("\nSuccessfully obtained the word links.")
    return word_links

print()
conn = connect()
if conn is not None:
    print("Successfully connected to PostgreSQL database.\n")
    all_words, word2id, id2word = load_data()
    data = dict()
    data["word_links"] = get_word_links(conn, all_words, word2id)

    print('\nDumping data in "word_links.pickle"...')
    with open("word_links.pickle", "wb") as f:
        pickle.dump(data, file=f)
    print('Successfully dumped data in "word_links.pickle".')

    print("\nClosing the connection to PostgreSQL database...")
    conn.close()
    print("Successfully closed the connection to PostgreSQL database.\n")