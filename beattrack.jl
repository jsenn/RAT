include("featurevecs.jl")
include("seq.jl")

using Statistics

@enum BTAlgo BT_symbolic BT_raw

function beattrack(rawaudio::SampleBuf; algo=BT_symbolic)
    if algo == BT_symbolic
        return btsymbolic(rawaudio)
    elseif algo == BT_raw
        return btraw(rawaudio)
    end
end

DEFAULT_QON = 0.9
DEFAULT_QOFF = 0.1
DEFAULT_BEATWIDTH = 60/(200*8)
DEFAULT_PERIOD = 60/100
DEFAULT_PHASE = 0
DEFAULT_VELOCITY = 0

struct BTSymbolicParams
    # Structural params
    qon::Float64        # probability of on-beat events.
    qoff::Float64       # probability of off-beat events.
    beatwidth::Float64  # stddev of the gaussian pulse that defines a beat.
                        # Default is the duration of a 32nd note at 200bpm.

    # Timing params
    period::Float64     # duration of a beat in seconds (default is 60/100bpm).
    phase::Float64      # duration from beginning of data to first beat (default 0).
    velocity::Float64   # change in period per second (default 0).
                        # Positive for accel; negative for rit.

    function BTSymbolicParams()
        new(
            DEFAULT_QON, DEFAULT_QOFF, DEFAULT_BEATWIDTH,
            DEFAULT_PERIOD, DEFAULT_PHASE, DEFAULT_VELOCITY
        )
    end

    function BTSymbolicParams(period::Number, phase::Number, velocity::Number)
        new(
            DEFAULT_QON, DEFAULT_QOFF, DEFAULT_BEATWIDTH,
            period, phase, velocity
        )
    end
end

"""
    btsymbolic(rawaudio::SampleBuf; blockduration=10, percentile=0.5, minbpm=40, maxbpm=200, bpmerr=0.05)

Implements a symbolic beat tracker as described in Rhythm and Transforms (p. 187).
"""
function btsymbolic(rawaudio::SampleBuf; blockduration=10, percentile=0.5, minbpm=40, maxbpm=200, bpmerr=0.05)
    features = btfeatures(rawaudio)
    @assert(!isempty(features) && all(feature -> length(feature) == length(features[1]), features))

    effsamplerate = features[1].samplerate
    @assert(all(feature -> feature.samplerate == effsamplerate, features))

    for feature in features
        tosymbolic!(feature; percentile=percentile)
    end

    blocksize = Int(ceil(blockduration * effsamplerate))
    blockoverlap = 0

    featuresplits = map(f -> arraysplit(f, blocksize, blockoverlap), features)
    numblocks = length(featuresplits[1])

    # Construct uniform grid of initial guesses (particles)
    secondlastbpm = (1-bpmerr) * maxbpm
    periodres = 60/secondlastbpm - 60/maxbpm
    minperiod = 60/maxbpm
    maxperiod = 60/minbpm
    periods = minperiod:periodres:maxperiod
    numperiods = length(periods)

    minphase = 0
    maxphase = maxperiod * 4 # wait as long as 4 beats at our slowest tempo for the first beat
    phaseres = 60 / (maxbpm*4) # determine phase to within 1 16th note at our fastest tempo
    phases = minphase:phaseres:maxphase
    numphases = length(phases)

    velocities = 0:0 # TODO

    guesses::Array{BTSymbolicParams,1} = []
    for period in periods
        for phase in phases
            if phase > period
                break
            end
            for velocity in velocities
                push!(guesses, BTSymbolicParams(period, phase, velocity))
            end
        end
    end

    function updateparams(oldparams::BTSymbolicParams)
        period_stddev = periodres
        phase_stddev = phaseres
        return BTSymbolicParams(
            gaussian(oldparams.period, period_stddev),
            gaussian(oldparams.phase, phase_stddev),
            0 # TODO
        )
    end

    function ll(params::BTSymbolicParams, data)
        return unknown_fixed_grid_seq_ll(data, effsamplerate, params.qon, params.qoff, params.period, params.phase, params.beatwidth)
    end

    function resample(guesses, weights)
        @assert(length(guesses) == length(weights))
        dist = Categorical(weights)
        new_guesses = []
        for i = 1:lastindex(guesses)
            push!(new_guesses, guesses[rand(dist)])
        end
        return new_guesses
    end

    for blockidx = 1:numblocks
        map!(updateparams, guesses, guesses)
        # TODO: use all feature vecs
        blockdata = featuresplits[1][blockidx]
        weights = map(g -> ll(g, blockdata), guesses)
        map!(exp, weights, weights)
        # normalize the weights
        weight_sum = sum(weights)
        if weight_sum > 0
            weights ./= weight_sum
        else
            fill!(weights, 1/length(weights))
        end
        guesses = resample(guesses, weights)
    end

    return guesses
end

function btraw(rawaudio::SampleBuf)
    @assert(false)
end

function btfeatures(rawaudio::SampleBuf)
    return [
        energyfeature(rawaudio),
        groupdelayfeature(rawaudio),
        spectralcenterfeature(rawaudio),
        spectraldispersionfeature(rawaudio)
    ]
end