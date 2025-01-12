import os
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from glossary import normalize_word


def statistics(iid2captions, iid2txt_ents, iid2split):
    all_images = {"train": [], "val": [], "test": []}
    all_texts = {"train": [], "val": [], "test": []}
    all_texts_per_image = {"train": [], "val": [], "test": []}
    all_txt_ents = {"train": [], "val": [], "test": []}
    for iid, texts in iid2captions.items():
        split = iid2split[iid]
        all_images[split].append(iid)
        all_texts_per_image[split].append(texts)
        all_texts[split].extend(texts)
        all_txt_ents[split].extend(iid2txt_ents[iid])

    for _dict in [all_images, all_texts, all_texts_per_image, all_txt_ents]:
        _dict["all"] = _dict["train"] + _dict["val"] + _dict["test"]

    for split, images in all_images.items():
        print(f"+ {split} set: {len(images)} images")

    for split, texts in all_texts.items():
        lengths = [len(text.split()) for text in texts]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {avg_len} words in average.")
        lengths = [length // 10 * 10 for length in lengths]
        print(Counter(lengths))

    for split, all_ents in all_txt_ents.items():
        lengths = [len(ents) for ents in all_ents]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {avg_len} entities in average.")

    for split, texts in all_texts_per_image.items():
        lengths = [len(texts_per_image) for texts_per_image in texts]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {avg_len} texts per images in average.")

    for split, texts in all_texts.items():
        lengths = [len(text.split(". ")) for text in texts]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {avg_len} sents in average.")


def path2rest(path, iid2captions, iid2img_ents, iid2txt_ents, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    img_ents = iid2img_ents[name]
    txt_ents = iid2txt_ents[name]
    split = iid2split[name]
    return [binary, captions, name, img_ents, txt_ents, split]


def make_arrow(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2img_ents = defaultdict(list)
    iid2txt_ents = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2img_ents[sample["img_path"]].extend(sample["image_entities"])
            iid2txt_ents[sample["img_path"]].extend(sample["text_entities"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2txt_ents, iid2split)
    bs = [path2rest(path, iid2captions, iid2img_ents, iid2txt_ents, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "img_ents", "txt_ents", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2img_ents, iid2txt_ents, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    chexpert = iid2chexpert[name]
    img_ents = iid2img_ents[name]
    txt_ents = iid2txt_ents[name]
    split = iid2split[name]
    return [binary, captions, name, chexpert, img_ents, txt_ents, split]


def make_arrow_mimic_cxr(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2img_ents = defaultdict(list)
    iid2txt_ents = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2img_ents[sample["img_path"]].extend(sample["image_entities"])
            iid2txt_ents[sample["img_path"]].extend(sample["text_entities"])
            iid2split[sample["img_path"]] = split

    caption_paths = []
    for path in tqdm(iid2captions):
        if os.path.exists(path):
            caption_paths.append(path)
        else:
            print(fr'NOT exist, {path}')

    print(f"+ {len(caption_paths)} images /  {len(iid2captions)} annotations")
    statistics(iid2captions, iid2txt_ents, iid2split)
    bs = [path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2img_ents, iid2txt_ents, iid2split)
          for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "img_ents",
                                                   "txt_ents", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def get_score(occurences):
    return 1.0


def path2rest_vqa(path, split, annotations, label2ans):
    with open(path, "rb") as fp:
        binary = fp.read()

    iid = path
    _annotation = annotations[split][iid]
    _annotation = list(_annotation.items())
    qids, qas = [int(a[0]) for a in _annotation], [a[1] for a in _annotation]
    questions = [qa[0] for qa in qas]
    img_ents = [img_ent for qa in qas for img_ent in qa[1]]
    img_ents = sorted(set(img_ents))
    txt_ents = [qa[2] for qa in qas]
    answers = [qa[3] for qa in qas]
    answer_labels = [a["labels"] for a in answers]
    answer_scores = [a["scores"] for a in answers]
    question_types = [a["answer_type"] for a in answers]
    answers = [[label2ans[l] for l in al] for al in answer_labels]
    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, question_types, 
                img_ents, txt_ents, split]

def make_arrow_vqa(data, dataset_name, save_dir):
    questions_train, questions_val, questions_test = data["train"], data["val"], data["test"]

    # Record Questions
    annotations = dict()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = defaultdict(dict)
        for q in tqdm(questions):
            _annotation[q["img_path"]][q["qid"]] = [q["question"], q["image_entities"], q["text_entities"]]
        annotations[split] = _annotation

    # Construct Vocabulary
    all_major_answers = list()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    print("Label size ({}): {}.".format(dataset_name, len(ans2label)))

    # Record Answers
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = annotations[split]
        for q in tqdm(questions):
            answers = normalize_word(str(q["answer"]).lower())
            answer_count = {}
            answer_count[answers] = answer_count.get(answers, 0) + 1
            labels = []
            scores = []
            for answer in answer_count:
                assert answer in ans2label
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)
            assert q['answer_type'].strip().lower() == "closed" or q['answer_type'].strip().lower() == "open"
            answer_type = 0 if q['answer_type'].strip().lower() == "closed" else 1
            _annotation[q["img_path"]][q["qid"]].append(
                {"labels": labels, "scores": scores, "answer_type": answer_type})

    # Write to the files
    for split in ["train", "val", "test"]:
        annot = annotations[split]
        annot_paths = [path for path in annot if os.path.exists(path)]
        assert len(annot_paths) == len(annot) or len(annot_paths) == len(annot) - 1
        print("{} set: {} images, {} questions".format(split,
                                                       len(annot),
                                                       len([vv for k, v in annot.items() for kk, vv in v.items()])))

        bs = [
            path2rest_vqa(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]
        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "answer_type",
                "img_ents",
                "txt_ents",
                "split",
            ],
        )
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_melinda(path, iid2captions, iid2i_meth, iid2p_meth, iid2i_meth_label, iid2p_meth_label, iid2img_ents,
                      iid2txt_ents, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    i_meth = iid2i_meth[name]
    p_meth = iid2p_meth[name]
    i_meth_label = iid2i_meth_label[name]
    p_meth_label = iid2p_meth_label[name]
    img_ents = iid2img_ents[name]
    txt_ents = iid2txt_ents[name]
    assert len(captions) == len(i_meth)
    assert len(captions) == len(p_meth)
    assert len(captions) == len(i_meth_label)
    assert len(captions) == len(p_meth_label)
    split = iid2split[name]
    return [binary, captions, name, i_meth, p_meth, i_meth_label, p_meth_label, img_ents, txt_ents, split]


def make_arrow_melinda(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2i_meth = defaultdict(list)
    iid2p_meth = defaultdict(list)
    iid2i_meth_label = defaultdict(list)
    iid2p_meth_label = defaultdict(list)
    iid2img_ents = defaultdict(list)
    iid2txt_ents = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split
            iid2i_meth[sample["img_path"]].append(sample["i_meth"])
            iid2p_meth[sample["img_path"]].append(sample["p_meth"])
            iid2i_meth_label[sample["img_path"]].append(sample["i_meth_label"])
            iid2p_meth_label[sample["img_path"]].append(sample["p_meth_label"])
            iid2img_ents[sample["img_path"]].extend(sample["image_entities"])
            iid2txt_ents[sample["img_path"]].extend(sample["text_entities"])

    i_meth_set = set([vv for k, v in iid2i_meth.items() for vv in v])
    i_meth_label_set = set([vv for k, v in iid2i_meth_label.items() for vv in v])
    p_meth_set = set([vv for k, v in iid2p_meth.items() for vv in v])
    p_meth_label_set = set([vv for k, v in iid2p_meth_label.items() for vv in v])

    i_meth_set = sorted(i_meth_set)
    i_meth_label_set = sorted(i_meth_label_set)
    p_meth_set = sorted(p_meth_set)
    p_meth_label_set = sorted(p_meth_label_set)

    i_meth_dict = {j: i for i, j in enumerate(i_meth_set)}
    p_meth_dict = {j: i for i, j in enumerate(p_meth_set)}
    i_meth_label_dict = {j: i for i, j in enumerate(i_meth_label_set)}
    p_meth_label_dict = {j: i for i, j in enumerate(p_meth_label_set)}

    iid2i_meth = {k: [i_meth_dict[vv] for vv in v] for k, v in iid2i_meth.items()}
    iid2p_meth = {k: [p_meth_dict[vv] for vv in v] for k, v in iid2p_meth.items()}
    iid2i_meth_label = {k: [i_meth_label_dict[vv] for vv in v] for k, v in iid2i_meth_label.items()}
    iid2p_meth_label = {k: [p_meth_label_dict[vv] for vv in v] for k, v in iid2p_meth_label.items()}

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2txt_ents, iid2split)
    bs = [path2rest_melinda(path, iid2captions, iid2i_meth, iid2p_meth, iid2i_meth_label, iid2p_meth_label,
                            iid2img_ents, iid2txt_ents, iid2split)
          for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "i_meth", "p_meth", "i_meth_label",
                                                   "p_meth_label", "img_ents", "txt_ents", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
