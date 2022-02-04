from al.bald.batch_bald import BatchBALDDropout
from al.kmeans import AntiKMeansSampler, KMeansSampler
from al.sampler import RandomSampler
from al.uncertainty import (
    AntiEntropySampler,
    AntiMarginSampler,
    EntropyDropoutSampler,
    EntropySampler,
    LeastConfidentDropoutSampler,
    LeastConfidentSampler,
    MarginDropoutSampler,
    MarginSampler,
    MostConfidentSampler,
)
from al.core_set import CoreSet
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
    CoreSet.name: CoreSet,
    BatchBALDDropout.name: BatchBALDDropout,
    MostConfidentSampler.name: MostConfidentSampler,
    AntiMarginSampler.name: AntiMarginSampler,
    AntiEntropySampler.name: AntiEntropySampler,
    AntiKMeansSampler.name: AntiKMeansSampler,
}


def get_al_sampler(key):
    return AL_SAMPLERS[key]
