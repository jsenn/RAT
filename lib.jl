using DSP
using PortAudio
using PyPlot
using QuadGK
using SampledSignals
using Statistics
using Unitful
using WAV

# =============================================================================
#                            AUDIO PRODUCTION
# =============================================================================

function make_white_noise(duration, samplerate)
    numsamples = Int(floor(duration * samplerate))
    data = 2 * rand(numsamples) .- 1
    return SampleBuf(data, samplerate)
end
	
# See https://dsp.stackexchange.com/a/17367
function make_chirp(startfreq, endfreq, duration, samplerate;
		    amplitude=1)
    starttime = 0s
    timestep = 1/samplerate
    endtime = duration - eps(Float64)s
    t = starttime:timestep:endtime

    omega1 = 2pi * startfreq
    omega2 = 2pi * endfreq

    data = amplitude * sin.(omega1*t + (omega2 - omega1)/duration * t.^2/2)

    return SampleBuf(data, samplerate)
end

function make_chord(frequencies::Array, duration, samplerate;
		    amplitude=1)
    starttime = 0s
    timestep = 1/samplerate
    endtime = duration - eps(Float64)s
    t = starttime:timestep:endtime

    data = zeros(length(t))
    for freq in frequencies
        data += amplitude * sin.(2pi * freq * t)
    end

    return SampleBuf(data, samplerate)
end

function make_var_freq(freqfunc, duration, samplerate; starttime=0s, amplitude=1)
    timestep = 1/samplerate
    endtime = duration - eps(Float64)s
    t = starttime:timestep:endtime

    integral = first.(quadgk.(freqfunc, starttime, t))
    data = amplitude * sin.(integral)

    return SampleBuf(data, samplerate)
end

function make_pause(duration, samplerate)
    data = zeros(Int(floor(samplerate * duration)))

    return SampleBuf(data, samplerate)
end

function upsample(sample::SampleBuf, newfreq)
    ratio = Int(floor(newfreq/sample.samplerate))
    @assert(ratio >= 2)
    data = zeros(ratio*length(sample.data))
    for i = 0:length(sample.data)-1
        for j = 0:ratio-1
            data[i*ratio + j + 1] = sample.data[i + 1]
        end
    end
    return SampleBuf(data, newfreq)
end

# =============================================================================
#                            AUDIO IO
# =============================================================================

function play(ostream::PortAudioStream, sample::SampleBuf)
    write(ostream, sample.data)
end

function play(sample::SampleBuf)
    ostream = PortAudioStream(0, 1)
    play(ostream, sample)
end

function save(sample::SampleBuf, filepath::String)
    wavwrite(sample.data, sample.samplerate, filepath)
end

function load_sample(filepath::String)
    signal, fps = wavread(filepath)
    signal = signal[:,1]

    return SampleBuf(signal, fps)
end

function listen(istream::PortAudioStream, duration)
    numframes = Int(floor(istream.samplerate * duration))
    buffer = read(istream, numframes)

    return SampleBuf(buffer.data[:,1], buffer.samplerate)
end

function listen(duration)
    istream = PortAudioStream(1, 0)

    return listen(istream, duration)
end

# =============================================================================
#                            AUDIO ANALYSIS
# =============================================================================
function bandpass(sample::SampleBuf, low, high)
    responsetype = Bandpass(low, high, fs=sample.samplerate)
    designmethod = Butterworth(4)

    filtered = filt(digitalfilter(responsetype, designmethod), sample.data)

    return SampleBuf(filtered, sample.samplerate)
end

function make_spectrogram(sample::SampleBuf;
			  timeslice=.001, sliceoverlap=.00025,
			  window=hamming)
    samples_per_slice = Int(floor(sample.samplerate * timeslice))
    sample_overlap = Int(floor(sample.samplerate * sliceoverlap))

    return spectrogram(sample.data, samples_per_slice, sample_overlap; window=window)
end

function plot_spectrogram(sample::SampleBuf;
			  timerange=:auto=>:auto, freqrange=:auto=>:auto,
			  timeslice=.001, sliceoverlap=.00025,
			  window=hamming)
    spect = make_spectrogram(sample, timeslice=timeslice, sliceoverlap=sliceoverlap, window=window)

    times = time(spect)/sample.samplerate

    # Sorted list of frequency buckets, normalized to the range [0, 0.5] (since
    # we can only recover frequencies as high as 0.5 * samplerate anyway)
    frequencies = freq(spect)

    # A 2D matrix where the entry in the ith row and jth column is the power level
    # of frequencies[i] during the time period starting at times[j]
    powers = log10.(power(spect))

    firsttime, lasttime = timerange
    firstfreq, lastfreq = freqrange
    firsttime = firsttime == :auto ? first(times) : firsttime
    lasttime = lasttime == :auto ? last(times) : lasttimed
    firstfreq = firstfreq == :auto ? first(frequencies) * sample.samplerate : firstfreq
    lastfreq = lastfreq == :auto ? last(frequencies) * sample.samplerate : lastfreq

    # Translate into image coords, with highest frequencies first
    image = reverse(powers, dims=1)
    freqidx(freq) = length(frequencies) - Int(floor((freq/sample.samplerate - first(frequencies)) / (last(frequencies) - first(frequencies)) * (length(frequencies) - 1)))
	timeidx(time) = 1 + Int(floor((time - first(times)) / (last(times) - first(times)) * (length(times) - 1)))
    imshow(image[freqidx(lastfreq):freqidx(firstfreq),timeidx(firsttime):timeidx(lasttime)], extent=[firsttime, lasttime, firstfreq, lastfreq], aspect="auto", cmap="binary")

    return spect
end
