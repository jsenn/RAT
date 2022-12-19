include("lib.jl")
include("util.jl")

using FFTW
using Polynomials

"""
    critband(sample::SampleBuf)

Calculate a series of feature vectors from the given `sample` containing the
amount of energy in each of many frequency bands. This is done in with a
rolling window, so that 

This is meant to mimic the function of the Basilar Membrane, with the bandpass
filters simulating the Critical Bands.

See Rhythm and Transforms p. 103-104
"""
function critband(sample::SampleBuf)
end

"""
    energyfeature(sample::SampleBuf; windowsize=0.01s, windowoverlap=0.005s)

Calculates the energy in windows of the given sample, returning the difference
between successive windows.

See Rhythm and Transforms p. 105-106
"""
function energyfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005, windowfunc=DSP.hanning)
    res = mapwindows(w -> sum(w.^2), sample, windowsize, windowoverlap; windowfunc=windowfunc)

    return SampleBuf(fdiff(res.data), res.samplerate)
end

"""
    groupdelayfeature(sapmle::SampleBuf; windowsize=0.01s, windowoverlap=0.005s)

TBW
"""
function groupdelayfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005, windowfunc=DSP.hanning)
    function processwindow(window)
        coeffs = rfft(window)
        phases = unwrap(map(angle, coeffs))
        # fit a line to the unwrapped phases, returning the slope
        poly = Polynomials.fit(1:lastindex(phases), phases, 1)
        return poly.coeffs[1]
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap; windowfunc=windowfunc)

    return res
end

function bin2freq(bin, samplerate, nfft)
    return bin * samplerate/nfft
end

function spectralcenter(xs::Array, samplerate::Number)
    #idx = weightedmedian(abs.(rfft(xs)))
    #return bin2freq(idx, samplerate, length(xs))
    mag = abs.(rfft(xs))
    frq = rfftfreq(length(xs), samplerate)
    return sum(frq .* mag) / sum(mag)
end

"""
    spectralcenterfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005)

Calculates the spectral center of windows of the given sample, returning the
difference between successive windows. The spectral center is defined as the
weighted median of the magnitudes of the FFT of the window.

See Rhythm and Transforms p. 106
"""
function spectralcenterfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005, windowfunc=DSP.hanning)
    function processwindow(window)
        return spectralcenter(window, sample.samplerate)
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap; windowfunc=windowfunc)

    return SampleBuf(fdiff(res.data), res.samplerate)
end

function spectraldispersion(xs::Array, samplerate::Number)
    spec = abs.(rfft(xs))
    freqs = rfftfreq(length(xs), samplerate)
    fc = bin2freq(weightedmedian(spec), samplerate, length(xs))
    return sum((spec .^ 2) .* abs.(freqs .- fc))
end

"""
    spectraldispersionfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005)

Calculates the spectral dispersion of windows of the given sample, returning the
difference between successive windows. The spectral dispersion is defined as the
weighted median of the magnitudes of the FFT of the window.

See Rhythm and Transforms p. 106
"""
function spectraldispersionfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005, windowfunc=DSP.hanning)
    function processwindow(window)
        return spectraldispersion(window, sample.samplerate)
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap; windowfunc=windowfunc)

    return SampleBuf(fdiff(res.data), res.samplerate)
end

function mapwindows(func, sample::SampleBuf, windowsize, windowoverlap; windowfunc=DSP.rect)::SampleBuf
    windowsize = Int(floor(sample.samplerate * windowsize))
    windowoverlap = Int(floor(sample.samplerate * windowoverlap))
    slicestep = windowsize - windowoverlap
    window = windowfunc(windowsize)
    as = arraysplit(sample.data, windowsize, windowoverlap)

    featurefreq = Int(floor(sample.samplerate / slicestep))
    featuredata = map(seg -> func(window .* seg), as)

    return SampleBuf(featuredata, featurefreq)
end

"""
    listenablefeature(feature::SampleBuf)

Upsample the given feature sample to audio rate (44100Hz), then modulate with
noise, in order to be able to listen to the feature.

See Rhythm and Transforms p. 104-105
"""
function listenablefeature(feature::SampleBuf)
    featureup = upsample(feature, 44100)
    featureup.data .*= 2 * rand(length(featureup.data)) .- 1
    return featureup
end

"""
    playfeature(feature::SampleBuf)

Play an audio signal that corresponds to the given feature vector.

See listenablefeature.
"""
function playfeature(feature::SampleBuf)
    play(listenablefeature(feature))
end