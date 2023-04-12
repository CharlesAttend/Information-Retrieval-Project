from torch import is_tensor
from torch.utils.data import Dataset
from ir_datasets.datasets.base import Dataset as irDataset
from sentence_transformers import InputExample


class IrDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        ir_dataset: irDataset,
        transform_query: callable = None,
        transform_doc: callable = None,
    ):
        """Base qui ne fonctionne pas, car je sais pas ce dont on a besoin yet
        Si c'est les qrels, les scoreddocs ou juste les doc.
        Pour moi c'est scoreddocs

        Parameters
        ----------
        ir_dataset : Dataset from ir_dataset
            _description_
        transform_query : callable, optional
            Doit supporter un batch en entrée, by default None
        transform_doc : callable, optional
            _description_, by default None
        """
        super().__init__()
        self.transform_query = transform_query
        self.transform_doc = transform_doc
        self.ir_dataset = ir_dataset
        self.qrels_iter = ir_dataset.qrels_iter()
        self.query_iter = ir_dataset.queries_iter()
        self.docs_iter = ir_dataset.docs_iter()

    def __len__(self):
        return len(self.scoreddocs_count())

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        doc_id, query_id, revelance = self.qrels_iter[idx]
        doc = self.docs_iter[doc_id]
        query = self.query_iter[query_id]

        if self.transform_query:
            query = self.transform_query(query)

        if self.transform_doc:
            doc = self.transform_doc(query)

        return InputExample(texts=[query, doc], label=revelance)
        # return doc, query, revelance

    def get_by_ids(self, idx):
        doc_id, query_id, revelance = self.qrels_iter[idx]
        return doc_id, query_id, revelance