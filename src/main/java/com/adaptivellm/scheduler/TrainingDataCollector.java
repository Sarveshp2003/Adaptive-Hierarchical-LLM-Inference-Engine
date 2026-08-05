package com.adaptivellm.scheduler;



import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



/**
 * Collects scheduler experiences.
 *
 */
public final class TrainingDataCollector {


    private final List<TrainingSample> samples =
            new ArrayList<>();

    private Path persistencePath;
    private BufferedWriter persistenceWriter;
    private boolean headerWritten;



    /**
     * Records decision.
     */
    public synchronized TrainingSample record(
            MemoryState state,
            Decision decision
    )
    {
        return record(state, decision, null);
    }



    /**
     * Records decision and feature vector.
     */
    public synchronized TrainingSample record(
            MemoryState state,
            Decision decision,
            double[] features
    )
    {

        TrainingSample sample =
                new TrainingSample(
                        state,
                        decision,
                        features
                );


        samples.add(
                sample
        );


        return sample;
    }





    /**
     * Enables persistence of records to a CSV file.
     */
    public synchronized void enablePersistence(Path outputPath) throws IOException {
        if (outputPath == null) {
            throw new IllegalArgumentException("outputPath cannot be null");
        }

        this.persistencePath = outputPath.toAbsolutePath();
        Files.createDirectories(this.persistencePath.getParent());
        this.persistenceWriter = Files.newBufferedWriter(
                this.persistencePath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE
        );
        this.headerWritten = false;
    }



    /**
     * Disables persistence.
     */
    public synchronized void disablePersistence() throws IOException {
        if (persistenceWriter != null) {
            persistenceWriter.close();
        }
        persistenceWriter = null;
        persistencePath = null;
        headerWritten = false;
    }



    /**
     * Updates result after execution.
     */
    public synchronized void updateResult(
            TrainingSample sample,
            double latencyImprovement,
            long memorySavedBytes
    )
    {

        sample.updateResult(
                latencyImprovement,
                memorySavedBytes
        );

        try {
            persistSample(sample);
        } catch (IOException e) {
            throw new RuntimeException("Failed to persist training sample", e);
        }
    }



    private void persistSample(TrainingSample sample) throws IOException {
        if (persistenceWriter == null) {
            return;
        }

        if (!headerWritten) {
            writeHeader();
            headerWritten = true;
        }

        MemoryState state = sample.state();
        Decision decision = sample.decision();
        double[] features = sample.features();

        StringBuilder line = new StringBuilder();
        line.append(sample.timestamp()).append(',');
        line.append(decision.action()).append(',');
        line.append(decision.targetId()).append(',');
        line.append(decision.confidence()).append(',');
        line.append(state.currentLayer()).append(',');
        line.append(state.currentToken()).append(',');
        line.append(state.gpuUsage()).append(',');
        line.append(state.ramUsage()).append(',');
        line.append(state.storageLatency()).append(',');
        line.append(state.cachedLayers()).append(',');
        line.append(state.kvPages()).append(',');
        line.append(state.pressureScore()).append(',');
        if (features != null) {
            for (double feature : features) {
                line.append(feature).append(',');
            }
        }
        line.append(sample.latencyImprovement()).append(',');
        line.append(sample.memorySavedBytes()).append('\n');

        persistenceWriter.write(line.toString());
        persistenceWriter.flush();
    }



    private void writeHeader() throws IOException {
        StringBuilder header = new StringBuilder();
        header.append("timestamp,action,targetId,confidence,currentLayer,currentToken,gpuUsage,ramUsage,storageLatency,cachedLayers,kvPages,pressureScore,");
        header.append("feature0,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,");
        header.append("latencyImprovement,memorySavedBytes\n");
        persistenceWriter.write(header.toString());
        persistenceWriter.flush();
    }



    /**
     * Returns dataset snapshot.
     */
    public List<TrainingSample> samples()
    {

        return Collections.unmodifiableList(
                samples
        );
    }




    /**
     * Dataset size.
     */
    public int size()
    {
        return samples.size();
    }



    /**
     * Clears collected data.
     */
    public synchronized void clear()
    {
        samples.clear();
    }



    public synchronized Path getPersistencePath() {
        return persistencePath;
    }
}