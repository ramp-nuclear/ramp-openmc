import openmc
from corecompute.query.powerquery import HeatingRateQuery
from multipledispatch import dispatch
from scipy.constants import eV


def power_tally(query: HeatingRateQuery) -> openmc.Tally:
    """
    return a tally to compute the power per neutron source
    """
    if query.method not in [
        "score heating",
        "score heating-local",
        "score kappa-fission",
        "score fission-q-prompt",
        "score fission-q-recoverable",
    ]:
        raise NotImplementedError(
            f"The method {query.method} for power computation is not implemented inthe openmc adapter"
        )
    tally = openmc.Tally()
    tally.scores = [query.method.split()[-1]]
    return tally


@dispatch(object, object, HeatingRateQuery, object)
def get_result_from_statepoint(tallies, _, query, __) -> dict:
    tally = tallies[query]
    return dict(method=query.method, heating_per_neutron_source=(tally.get_values(value="mean").flatten()[0] * eV))
