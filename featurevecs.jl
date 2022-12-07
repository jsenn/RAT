include("lib.jl")
using FFTW
using Polynomials

"""
    critband(sample::Sample)

Calculate a series of feature vectors from the given `sample` containing the
amount of energy in each of many frequency bands. This is done in with a
rolling window, so that 

This is meant to mimic the function of the Basilar Membrane, with the bandpass
filters simulating the Critical Bands.

See Rhythm and Transforms p. 103-104
"""
function critband(sample::Sample)
end

"""
    energyfeature(sample::Sample; windowsize=0.01s, windowoverlap=0.005s)

Calculates the energy in windows of the given sample, returning the difference
between successive windows.

See Rhythm and Transforms p. 105-106
"""
function energyfeature(sample::Sample; windowsize=0.01s, windowoverlap=0.005s)
    res = mapwindows(w -> sum(w.^2), sample, windowsize, windowoverlap)

    return Sample(res.samplefreq, fdiff(res.data))
end

"""
    groupdelay(sapmle::Sample; windowsize=0.01s, windowoverlap=0.005s)

TBW
"""
function groupdelay(sample::Sample; windowsize=0.01s, windowoverlap=0.005s)
    function processwindow(window)
        coeffs = fft(window)
        unwrap!(map!(angle, coeffs, coeffs))
        # fit a line to the unwrapped phases, returning the slope
        poly = Polynomials.fit(1:length(coeffs), coeffs, 1)
        return poly.coeffs[1]
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap)

    return res
end

function mapwindows(func, sample::Sample, windowsize, windowoverlap)::Sample
    samples_per_slice = Int(floor(sample.samplefreq * windowsize))
    sample_overlap = Int(floor(sample.samplefreq * windowoverlap))
    slicestep = samples_per_slice - sample_overlap
    slicecount = Int(floor((length(sample.data) - sample_overlap) / slicestep))

    featurefreq = Int(floor(ustrip(sample.samplefreq) / slicestep))*Hz
    featuredata = zeros(slicecount)

    for widx = 1:slicecount
        woff = widx - 1
        wstart = 1 + woff * slicestep
        window = view(sample.data, wstart:wstart+samples_per_slice)
        featuredata[widx] = func(window)
    end

    return Sample(featurefreq, featuredata)
end

function fdiff(xs::Array)
    if length(xs) < 2
        return []
    end

    res = zeros(eltype(xs), length(xs) - 1)
    xcurr = xs[1]
    for i = 2:length(xs)
        res[i-1] = xs[i] - xcurr
        xcurr = xs[i]
    end

    return res
end

"""
    listenablefeature(feature::Sample)

Upsample the given feature sample to audio rates (44100Hz), then modulate with
noise, in order to be able to listen to the feature.

See Rhythm and Transforms p. 104-105
"""
function listenablefeature(feature::Sample)
    featureup = upsample(feature, 44100Hz)
    featureup.data .*= 2 * rand(length(featureup.data)) .- 1
    return featureup
end

"""
    playfeature(feature::Sample)

Play an audio signal that corresponds to the given feature vector.

See listenablefeature.
"""
function playfeature(feature::Sample)
    play(listenablefeature(feature))
end