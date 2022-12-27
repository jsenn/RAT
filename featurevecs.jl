include("lib.jl")
include("util.jl")

using FFTW
using Polynomials

"""
    energyfeature(sample::SampleBuf; windowsize=0.01s, windowoverlap=0.005s)

Calculates the energy in windows of the given sample, returning the difference
between successive windows.

See Rhythm and Transforms p. 105-106
"""
function energyfeature(sample::SampleBuf; windowsize=1024, windowoverlap=div(windowsize,2), windowfunc=DSP.hanning)
    res = mapwindows(w -> sum(w.^2), sample, windowsize, windowoverlap; windowfunc=windowfunc)

    normalize!(res.data)
    diff = fdiff(res.data)

    return SampleBuf(diff, res.samplerate)
end

"""
    groupdelayfeature(sapmle::SampleBuf; windowsize=0.01s, windowoverlap=0.005s)

TBW
"""
function groupdelayfeature(sample::SampleBuf; windowsize=1024, windowoverlap=div(windowsize,2), windowfunc=DSP.hanning)
    function processwindow(window)
        coeffs = rfft(window)
        freqs = rfftfreq(length(window), sample.samplerate)
        phases = unwrap(map(angle, coeffs))
        # fit a line to the unwrapped phases, returning the slope
        poly = Polynomials.fit(freqs, phases, 1)
        return poly.coeffs[2]
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap; windowfunc=windowfunc)

    normalize!(res.data)
    diff = fdiff(res.data)

    return SampleBuf(diff, res.samplerate)
end

function bin2freq(bin, samplerate, nfft)
    return bin * samplerate/nfft
end

function freq2bin(freq, samplerate, nfft)
    return Int(round(freq * nfft / samplerate))
end

function spectralcenter(xs::Array, samplerate::Number)
    idx = weightedmedian(abs.(rfft(xs)))
    return bin2freq(idx, samplerate, length(xs))
end

"""
    spectralcenterfeature(sample::SampleBuf; windowsize=0.01, windowoverlap=0.005)

Calculates the spectral center of windows of the given sample, returning the
difference between successive windows. The spectral center is defined as the
weighted median of the magnitudes of the FFT of the window.

See Rhythm and Transforms p. 106
"""
function spectralcenterfeature(sample::SampleBuf; windowsize=1024, windowoverlap=div(windowsize,2), windowfunc=DSP.hanning)
    function processwindow(window)
        return spectralcenter(window, sample.samplerate)
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap; windowfunc=windowfunc)

    normalize!(res.data)
    diff = fdiff(res.data)

    return SampleBuf(diff, res.samplerate)
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
function spectraldispersionfeature(sample::SampleBuf; windowsize=1024, windowoverlap=div(windowsize,2), windowfunc=DSP.hanning)
    function processwindow(window)
        return spectraldispersion(window, sample.samplerate)
    end
    res = mapwindows(processwindow, sample, windowsize, windowoverlap; windowfunc=windowfunc)

    normalize!(res.data)
    diff = fdiff(res.data)

    return SampleBuf(diff, res.samplerate)
end

function mapwindows(func, sample::SampleBuf, windowsize, windowoverlap; windowfunc=DSP.rect)::SampleBuf
    slicestep = windowsize - windowoverlap
    window = windowfunc(windowsize)
    as = arraysplit(sample.data, windowsize, windowoverlap)

    featurefreq = Int(floor(sample.samplerate / slicestep))
    featuredata = map(seg -> func(window .* seg), as)

    return SampleBuf(featuredata, featurefreq)
end

"""
    listenablefeature(feature::SampleBuf; samplerate=44100)

Upsample the given feature sample to audio rate (default 44100Hz), then
modulate with noise, in order to be able to listen to the feature.

See Rhythm and Transforms p. 104-105
"""
function listenablefeature(feature::SampleBuf; samplerate=44100)
    featureup = upsample(feature, samplerate)
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