from al.kmeans import KMeansSampler
from al.sampler import RandomSampler
from al.uncertainty import EntropySampler, LeastConfidentSampler, MarginSampler


AL_SAMPLERS = {
    RandomSampler.name: RandomSampler,
    LeastConfidentSampler.name: LeastConfidentSampler,
    MarginSampler.name: MarginSampler,
    EntropySampler.name: EntropySampler,
    KMeansSampler.name: KMeansSampler,
}


def get_al_sampler(key):
    return AL_SAMPLERS[key]
