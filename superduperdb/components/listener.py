import dataclasses as dc
import typing as t

from overrides import override

from superduperdb import CFG
from superduperdb.backends.base.query import CompoundSelect
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import _OUTPUTS_KEY
from superduperdb.base.serializable import Variable
from superduperdb.components.model import Mapping
from superduperdb.misc.annotations import public_api
from superduperdb.misc.server import request_server

from ..jobs.job import Job
from .component import Component
from .model import ModelInputType, _Predictor


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class Listener(Component):
    """
    Listener object which is used to process a column/ key of a collection or table,
    and store the outputs.
    {component_parameters}
    :param key: Key to be bound to model
    :param model: Model for processing data
    :param select: Object for selecting which data is processed
    :param identifier: A string used to identify the model.
    :param active: Toggle to ``False`` to deactivate change data triggering
    :param predict_kwargs: Keyword arguments to self.model.predict
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    key: ModelInputType
    model: _Predictor
    select: CompoundSelect
    active: bool = True
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    identifier: str = ''

    type_id: t.ClassVar[str] = 'listener'

    def __post_init__(self, artifacts):
        if self.identifier == '':
            self.identifier = f'{self.id_key}::{self.model.identifier}::{self.model.version}'
        super().__post_init__(artifacts)

    @property
    def mapping(self):
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        if self.model.version is not None:
            return (
                f'{_OUTPUTS_KEY}.{self.identifier}.{self.model.version}'
            )
        else:
            def _callback(db, value, kwargs):
                return value.format(version=self.version)
            return Variable(
                f'{_OUTPUTS_KEY}.{self.identifier}.{"{version}"}',
                setter_callback=_callback
            )

    @override
    def pre_create(self, db: Datalayer) -> None:
        if isinstance(self.model, str):
            self.model = t.cast(_Predictor, db.load('model', self.model))

        if self.select is not None and self.select.variables:
            self.select = t.cast(CompoundSelect, self.select.set_variables(db))

        # This logic intended to format the version of upstream `Listener`
        # instances into `self.key`, so that the `Listener` knows exactly where
        # to find the `_outputs` subfield: `_outputs.<identifier>.<version>`
        # of a `Listener`.
        self.key = self._set_key(db, self.key)

    @staticmethod
    def _set_key(db, key, **kwargs):
        if isinstance(key, Variable):
            return key.set(db, **kwargs)
        elif isinstance(key, (list, tuple)):
            return [Listener._set_key(db, x, **kwargs) for x in key]
        elif isinstance(key, dict):
            return {k: Listener._set_key(db, v, **kwargs) for k, v in key.items()}
        return key

    @override
    def post_create(self, db: Datalayer) -> None:
        # Start cdc service if enabled
        if self.select is not None and self.active and not db.server_mode:
            if CFG.cluster.cdc.uri:
                request_server(
                    service='cdc',
                    endpoint='listener/add',
                    args={'name': self.identifier},
                    type='get',
                )
            else:
                db.cdc.add(self)

    @property
    def dependencies(self) -> t.List[str]:
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        all_ = [
            x.value if isinstance(x, Variable) else x for x in all_
        ]
        return [
            x.split('.')[1]
            for x in all_
            if x.startswith(f'_outputs')
        ]

    @property
    def id_key(self) -> str:
        def _id_key(key) -> str:
            if isinstance(key, str):
                if key.startswith('_outputs.'):
                    return key.split('.')[1]
                else:
                    return key
            elif isinstance(key, (tuple, list)):
                return ','.join([_id_key(k) for k in key])
            elif isinstance(key, dict):
                return ','.join([_id_key(k) for k in key.values()])
            else:
                raise TypeError('Type of key is not valid')

        return _id_key(self.key)

    def depends_on(self, other: Component):
        if not isinstance(other, Listener):
            return False

        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())

        return any([
            x.startswith(f'_outputs.{other.identifier}')
            for x in all_
        ])

    @override
    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
    ) -> t.Sequence[t.Any]:
        """
        Schedule jobs for the listener

        :param database: The DB instance to process
        :param dependencies: A list of dependencies
        :param verbose: Whether to print verbose output
        """
        if not self.active:
            return []
        assert not isinstance(self.model, str)

        out = [
            self.model.predict_in_db_job(
                X=self.key,
                db=db,
                select=self.select.copy(),
                dependencies=dependencies,
            )
        ]
        return out

    def cleanup(self, database: Datalayer) -> None:
        """Clean up when the listener is deleted

        :param database: The DB instance to process
        """
        # TODO - this doesn't seem to do anything
        if (cleanup := getattr(self.select, 'model_cleanup', None)) is not None:
            assert not isinstance(self.model, str)
            cleanup(database, model=self.model.identifier, key=self.key)
