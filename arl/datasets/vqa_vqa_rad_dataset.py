from .base_dataset import BaseDataset


class VQAVQARADDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqa_vqa_rad_train"]
        elif split == "val":
            names = ["vqa_vqa_rad_val"]
        elif split == "test":
            names = ["vqa_vqa_rad_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions"
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        txt = self.get_text(index)
        text = txt["text"]
        img_label = txt["img_label"]
        txt_label = txt["txt_label"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        answers = self.table["answers"][index][question_index].as_py()
        labels = self.table["answer_labels"][index][question_index].as_py()
        scores = self.table["answer_scores"][index][question_index].as_py()
        answer_types = self.table["answer_type"][index][question_index].as_py()

        # print('->' * 30)
        # print(fr'text: {text}')
        # print(fr'vqa_answer: {answers}')
        # print(fr'vqa_labels: {labels}')
        # print(fr'answer_types: {answer_types}')
        # print('->' * 30)

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "answer_types": answer_types,
            "qid": qid,
            "img_label": img_label,
            "txt_label": txt_label
        }
