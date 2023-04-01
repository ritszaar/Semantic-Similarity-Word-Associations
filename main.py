import re
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
    print("Words saved in ./Words/all_words.txt.\n")
    return all_words, word2id, id2word

def get_image_links(word2id):
    image_links = []
    for i in range(0, 50000, 1000):
        with open("./Base Associations/{}-{}.txt".format(i, i + 1000 - 1), "r") as f:
            for i in range(0, 11000, 11):
                line = f.readline()
                id = int(line.split(" ")[0])
                fine_label_id = word2id[re.findall('"([^"]*)"', line)[0]]
                image_links.append([])
                image_links[id].append((fine_label_id, 1.00))
                for i in range(10):
                    if i < topIWK:
                        line = f.readline()
                        strength = float(line.split(" ")[-1])
                        pred_label_id = word2id[re.findall('"([^"]*)"', line)[0]]
                        image_links[id].append((pred_label_id, strength))
                    else:
                        f.readline()
    return image_links

def get_word_links(conn, all_words, word2id):
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
                print("Cue: {}, Target: {}".format(all_words[i], all_words[j]))
            cursor.close() 
            count = count + 1
            if count % 1000 == 0:
                print("Found word strengths ({}/{})".format(count, total_count))
    return word_links

print()
conn = connect()
if conn is not None:
    print("Successfully connected to PostgreSQL database.\n")
    all_words, word2id, id2word = get_all_words(conn)
    image_links = get_image_links(word2id)
    word_links = get_word_links(conn, all_words, word2id)
    for i in range(len(all_words)):
        for j in range(len(all_words)):
            if word_links[i][j] != -1:
                print("Cue: {}, Target: {}".format(id2word[i], id2word[j]))
