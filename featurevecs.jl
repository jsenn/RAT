include("lib.jl")

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
    samples_per_slice = Int(floor(sample.samplefreq * windowsize))
    sample_overlap = Int(floor(sample.samplefreq * windowoverlap))
    slicestep = samples_per_slice - sample_overlap

    slicecount = Int(floor((length(sample.data) - sample_overlap) / slicestep))
    featurefreq = Int(floor(ustrip(sample.samplefreq) / slicestep))*Hz
    featuredata = zeros(slicecount - 1) # - 1 because we take differences
    
    firstwindow = view(sample.data, 1:samples_per_slice+1)
    currenergy = sum(firstwindow .^ 2)
    for widx = 1:slicecount - 1
        wstart = 1 + widx * slicestep
        window = view(sample.data, wstart:wstart+samples_per_slice)
        energy = sum(window.^2)
        featuredata[widx] = energy - currenergy
        currenergy = energy
    end

    return Sample(featurefreq, featuredata)
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