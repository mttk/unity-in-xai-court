from al.kmeans import KMeansSampler
from al.sampler import RandomSampler
from al.uncertainty import (
    EntropyDropoutSampler,
    EntropySampler,
    LeastConfidentDropoutSampler,
    LeastConfidentSampler,
    MarginDropoutSampler,
    MarginSampler,
)
from al.badge import BADGE


AL_SAMPLERS = {
    RandomSampler.name: RandomSampler,
    LeastConfidentSampler.name: LeastConfidentSampler,
    MarginSampler.name: MarginSampler,
    EntropySampler.name: EntropySampler,
    KMeansSampler.name: KMeansSampler,
    LeastConfidentDropoutSampler.name: LeastConfidentDropoutSampler,
    MarginDropoutSampler.name: MarginDropoutSampler,
    EntropyDropoutSampler.name: EntropyDropoutSampler,
    BADGE.name: BADGE,
}


def get_al_sampler(key):
    return AL_SAMPLERS[key]