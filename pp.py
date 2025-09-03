def read_and_preprocess_documents(directory_path):
    documents_list = []
    doc_id_counter = 1

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Split by numbers followed by a dot (1., 2., etc.)
                docs = re.split(r'\n\d+\.\s*', content.strip())
                docs = [doc.strip() for doc in docs if doc.strip()]  

                for doc in docs:
                    documents_list.append({
                        'doc_id': doc_id_counter,
                        'filename': filename,
                        'text': doc
                    })
                    doc_id_counter += 1

    df = pd.DataFrame(documents_list)
    df['processed_text'] = df['text'].apply(basic_preprocess_text)
    return df
