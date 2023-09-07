import dataclasses as dc
import typing as t
from functools import cached_property

from superduperdb import CFG
from superduperdb.container.document import Document
from superduperdb.container.encoder import Encoder
from superduperdb.misc.files import load_uris
from superduperdb.misc.special_dicts import MongoStyleDict


@dc.dataclass
class SuperDuperCursor:
    """
    A cursor that wraps a cursor and returns ``Document`` wrapping
    a dict including ``Encodable`` objects.
    """

    #: the cursor to wrap
    raw_cursor: t.Any

    #: the field to use as the document id
    id_field: str

    #: a dict of encoders to use to decode the documents
    encoders: t.Dict[str, Encoder] = dc.field(default_factory=dict)

    #: a dict of features to add to the documents
    features: t.Optional[t.Dict[str, str]] = None

    #: a dict of scores to add to the documents
    scores: t.Optional[t.Dict[str, float]] = None

    _it: int = 0

    @cached_property
    def _results(self) -> t.Optional[t.List[t.Dict]]:
        def key(r):
            return -self.scores[str(r[self.id_field])]

        return None if self.scores is None else sorted(self.raw_cursor, key=key)

    @staticmethod
    def add_features(r, features):
        """
        Add features to a document.

        :param r: the document
        :param features: the features to add
        """
        r = MongoStyleDict(r)
        for k in features:
            r[k] = r['_outputs'][k][features[k]]
        if '_other' in r:
            for k in features:
                if k in r['_other']:
                    r['_other'][k] = r['_outputs'][k][features[k]]
        return r

    def count(self):
        """
        Count the number of documents in the cursor.
        """
        return len(list(self.raw_cursor.clone()))

    def limit(self, *args, **kwargs):
        """
        Limit the number of documents in the cursor.
        """
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            encoders=self.encoders,
            features=self.features,
            scores=self.scores,
        )

    @staticmethod
    def wrap_document(r, encoders):
        """
        Wrap a document in a ``Document``.
        """
        return Document(Document.decode(r, encoders))

    def __iter__(self):
        return self

    def cursor_next(self):
        return self.raw_cursor.next()

    def __next__(self):
        if self.scores is not None:
            try:
                r = self._results[self._it]
            except IndexError:
                raise StopIteration
            self._it += 1
        else:
            r = self.cursor_next()
        if self.scores is not None:
            r['_score'] = self.scores[str(r[self.id_field])]
        if self.features is not None and self.features:
            r = self.add_features(r, features=self.features)
        if CFG.downloads.hybrid:
            load_uris(r, CFG.downloads.root)
        return self.wrap_document(r, self.encoders)

    next = __next__
