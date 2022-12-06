using Statistics

using DSP
using QuadGK
using PortAudio
using PyPlot
using Unitful
using WAV

import Unitful: Area, Frequency, Time, Length, Velocity, Hz, m, s, rad, °

const Angle{T} = Union{Quantity{T, NoDims, typeof(rad)}, Quantity{T, NoDims, typeof(°)}}

# =============================================================================
#                            AUDIO PRODUCTION
# =============================================================================

struct Sample
    samplefreq::Frequency
    data::Array
end

function make_white_noise(duration::Time, samplefreq::Frequency)
    numsamples = Int(floor(duration * samplefreq))
    data = 2 * rand(numsamples) .- 1
    return Sample(samplefreq, data)
end
	
# See https://dsp.stackexchange.com/a/17367
function make_chirp(startfreq::Frequency, endfreq::Frequency, duration::Time, samplefreq::Frequency;
		    amplitude=1)
    starttime = 0s
    timestep = 1/samplefreq
    endtime = duration - eps(Float64)s
    t = starttime:timestep:endtime

    omega1 = 2pi * startfreq
    omega2 = 2pi * endfreq

    data = amplitude * sin.(omega1*t + (omega2 - omega1)/duration * t.^2/2)

    return Sample(samplefreq, data)
end

function make_chord(frequencies::Array, duration::Time, samplefreq::Frequency;
		    amplitude=1)
    starttime = 0s
    timestep = 1/samplefreq
    endtime = duration - eps(Float64)s
    t = starttime:timestep:endtime

    data = zeros(length(t))
    for freq in frequencies
        data += amplitude * sin.(2pi * freq * t)
    end

    return Sample(samplefreq, data)
end

function make_var_freq(freqfunc, duration::Time, samplefreq::Frequency; starttime=0s, amplitude=1)
    timestep = 1/samplefreq
    endtime = duration - eps(Float64)s
    t = starttime:timestep:endtime

    integral = first.(quadgk.(freqfunc, starttime, t))
    data = amplitude * sin.(integral)

    return Sample(samplefreq, data)
end

function make_pause(duration::Time, samplefreq::Frequency)
    data = zeros(Int(floor(samplefreq * duration)))

    return Sample(samplefreq, data)
end

function upsample(sample::Sample, newfreq::Frequency)
    ratio = Int(floor(newfreq/sample.samplefreq))
    @assert(ratio >= 2)
    data = zeros(ratio*length(sample.data))
    for i = 0:length(sample.data)-1
        for j = 0:ratio-1
            data[i*ratio + j + 1] = sample.data[i + 1]
        end
    end
    return Sample(newfreq, data)
end

# =============================================================================
#                            AUDIO IO
# =============================================================================

function play(ostream::PortAudioStream, sample::Sample)
    write(ostream, sample.data)
end

function play(sample::Sample)
    ostream = PortAudioStream(0, 1)
    play(ostream, sample)
end

function save(sample::Sample, filepath::String)
    wavwrite(sample.data, ustrip(uconvert(Hz, sample.samplefreq)), filepath)
end

function load_sample(filepath::String)
    signal, fps = wavread(filepath)
    signal = signal[:,1]

    return Sample(fps*Hz, signal)
end

function listen(istream::PortAudioStream, duration::Time)
    numframes = Int(floor(istream.samplerate*Hz * uconvert(s, duration)))
    buffer = read(istream, numframes)

    return Sample(buffer.samplerate*Hz, buffer.data[:,1])
end

function listen(duration::Time)
    istream = PortAudioStream(1, 0)

    return listen(istream, duration)
end

# =============================================================================
#                            AUDIO ANALYSIS
# =============================================================================
function bandpass(sample::Sample, low::Frequency, high::Frequency)
    low = ustrip(uconvert(Hz, low))
    high = ustrip(uconvert(Hz, high))
    fs = ustrip(uconvert(Hz, sample.samplefreq))
    responsetype = Bandpass(low, high, fs=fs)
    designmethod = Butterworth(4)

    filtered = filt(digitalfilter(responsetype, designmethod), sample.data)

    return Sample(sample.samplefreq, filtered)
end

function make_spectrogram(sample::Sample;
			  timeslice::Time=.001s, sliceoverlap::Time=.00025s,
			  window=hamming)
    samples_per_slice = Int(floor(sample.samplefreq * timeslice))
    sample_overlap = Int(floor(sample.samplefreq * sliceoverlap))

    return spectrogram(sample.data, samples_per_slice, sample_overlap; window=window)
end

function plot_spectrogram(sample::Sample;
			  timerange=:auto=>:auto, freqrange=:auto=>:auto,
			  timeslice::Time=.001s, sliceoverlap::Time=.00025s,
			  window=hamming)
    spect = make_spectrogram(sample, timeslice=timeslice, sliceoverlap=sliceoverlap, window=window)

    times = ustrip.(time(spect)/sample.samplefreq)

    # Sorted list of frequency buckets, normalized to the range [0, 0.5] (since
    # we can only recover frequencies as high as 0.5 * samplefreq anyway)
    frequencies = freq(spect)

    # A 2D matrix where the entry in the ith row and jth column is the power level
    # of frequencies[i] during the time period starting at times[j]
    powers = log10.(power(spect))

    firsttime, lasttime = timerange
    firstfreq, lastfreq = freqrange
    firsttime = firsttime == :auto ? first(times) : firsttime
    lasttime = lasttime == :auto ? last(times) : lasttimed
    firstfreq = firstfreq == :auto ? first(frequencies) * ustrip(sample.samplefreq) : firstfreq
    lastfreq = lastfreq == :auto ? last(frequencies) * ustrip(sample.samplefreq) : lastfreq

    # Translate into image coords, with highest frequencies first
    image = reverse(powers, dims=1)
    freqidx(freq) = length(frequencies) - Int(floor((freq/ustrip(sample.samplefreq) - first(frequencies)) / (last(frequencies) - first(frequencies)) * (length(frequencies) - 1)))
	timeidx(time) = 1 + Int(floor((time - first(times)) / (last(times) - first(times)) * (length(times) - 1)))
    imshow(image[freqidx(lastfreq):freqidx(firstfreq),timeidx(firsttime):timeidx(lasttime)], extent=[firsttime, lasttime, firstfreq, lastfreq], aspect="auto", cmap="binary")

    return spect
end
