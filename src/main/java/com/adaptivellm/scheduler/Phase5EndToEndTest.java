package com.adaptivellm.scheduler;

import java.io.IOException;
import java.util.*;

/**
 * Phase 5 comprehensive integration tests.
 * 
 * Tests:
 * - Neural network predictions
 * - Model training and evaluation
 * - Model persistence
 * - Hyperparameter optimization
 * - Cross-validation
 * - Performance optimization
 * - End-to-end scheduling
 */
public final class Phase5EndToEndTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║         PHASE 5: AI SCHEDULER END-TO-END TESTS             ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        try {
            testNeuralNetworkPredictor();
            testModelPersistence();
            testMLTrainer();
            testPerformanceOptimizer();
            testSchedulerIntegration();
            testHyperparameterOptimization();
            testCrossValidation();

            System.out.println("\n╔════════════════════════════════════════════════════════════╗");
            System.out.println("║              TEST SUMMARY                                  ║");
            System.out.println("╚════════════════════════════════════════════════════════════╝");
            System.out.println("PASSED: " + passed);
            System.out.println("FAILED: " + failed);
            System.out.println("TOTAL:  " + (passed + failed));

            if (failed == 0) {
                System.out.println("\n✅ ALL TESTS PASSED!\n");
            } else {
                System.out.println("\n❌ SOME TESTS FAILED\n");
                System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Test execution failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testNeuralNetworkPredictor() {
        System.out.println("\n[TEST 1] Neural Network Predictor");
        System.out.println("─".repeat(50));

        try {
            NeuralNetworkPredictor nn = new NeuralNetworkPredictor();
            
            // Test prediction with normalized input derived from MemoryState
            FeatureExtractor extractor = new FeatureExtractor();
            MemoryState state = new MemoryState(
                10,    // current_layer
                1000,  // current_token
                0.7,   // gpu_usage
                0.6,   // ram_usage
                50.0,  // storage_latency
                5,     // cached_layers
                20     // kv_pages
            );
            double[] testFeatures = extractor.extractNormalized(state);


            Decision prediction = nn.predict(testFeatures);
            assert prediction.action() != null : "Prediction returned null action";
            assert prediction.confidence() >= 0.0 && prediction.confidence() <= 1.0 : "Invalid confidence";

            System.out.println("✓ Network initialization");
            System.out.println("✓ Prediction: " + prediction);
            System.out.println("✓ Network info: " + nn.getNetworkInfo());
            System.out.println("✓ Statistics: " + nn.getStatistics());

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            failed++;
        }
    }

    private static void testModelPersistence() {
        System.out.println("\n[TEST 2] Model Persistence");
        System.out.println("─".repeat(50));

        try {
            String modelDir = "test_models";
            ModelPersistence persistence = new ModelPersistence(modelDir);
            NeuralNetworkPredictor model = new NeuralNetworkPredictor();

            // Save model
            String modelName = persistence.saveModel(model, "test_model", 100, 0.5);
            System.out.println("✓ Model saved: " + modelName);

            // Load model
            NeuralNetworkPredictor loaded = persistence.loadModel(modelName);
            System.out.println("✓ Model loaded");

            // List models
            List<String> models = persistence.listModels();
            assert !models.isEmpty() : "No models found";
            System.out.println("✓ Models listed: " + models.size() + " found");

            // Get metadata
            Map<String, String> metadata = persistence.getModelMetadata(modelName);
            assert metadata.containsKey("samples_used") : "Missing metadata";
            System.out.println("✓ Metadata retrieved: " + metadata.size() + " fields");

            // Validate model
            boolean isValid = persistence.validateModel(modelName);
            assert isValid : "Model validation failed";
            System.out.println("✓ Model validation passed");

            // Generate report
            String report = persistence.generateReport();
            assert report.contains(modelName) : "Report missing model";
            System.out.println("✓ Report generated");

            // Cleanup
            persistence.deleteModel(modelName);
            System.out.println("✓ Model deleted");

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            failed++;
        }
    }

    private static void testMLTrainer() {
        System.out.println("\n[TEST 3] ML Trainer");
        System.out.println("─".repeat(50));

        try {
            NeuralNetworkPredictor model = new NeuralNetworkPredictor();
            ModelPersistence persistence = new ModelPersistence("test_models");
            MLTrainer trainer = new MLTrainer(model, persistence);

            // Generate synthetic training data
            List<TrainingSample> samples = generateSyntheticData(200);

            // Train model
            trainer.train(samples);
            System.out.println("✓ Model trained on " + samples.size() + " samples");

            // Get metrics
            Map<String, Double> metrics = trainer.getMetrics();
            assert metrics.containsKey("accuracy") : "Missing accuracy metric";
            System.out.println("✓ Metrics: " + metrics);

            // Get report
            String report = trainer.getReport();
            assert report.contains("Performance Metrics") : "Invalid report";
            System.out.println("✓ Report generated");

            // Compare with baseline
            trainer.compareWithBaseline(samples);
            System.out.println("✓ Baseline comparison done");

            // Get action distribution
            Map<SchedulerAction, Integer> dist = trainer.getActionDistribution();
            assert !dist.isEmpty() : "Empty action distribution";
            System.out.println("✓ Action distribution: " + dist.size() + " actions");

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }

    private static void testPerformanceOptimizer() {
        System.out.println("\n[TEST 4] Performance Optimizer");
        System.out.println("─".repeat(50));

        try {
            FeatureExtractor extractor = new FeatureExtractor();
            TrainingDataCollector collector = new TrainingDataCollector();
            AdaptiveScheduler scheduler = new AdaptiveScheduler(extractor, 
                new RuleBasedPredictor(), collector);
            
            PerformanceOptimizer optimizer = new PerformanceOptimizer(scheduler, collector);

            // Get default config
            PerformanceOptimizer.OptimizationConfig config = optimizer.getDefaultConfig();
            assert config != null : "Config is null";
            System.out.println("✓ Default config: " + config);

            // Analyze different profiles
            PerformanceOptimizer.OptimizationConfig aggressive = 
                optimizer.analyze(PerformanceOptimizer.OptimizationProfile.AGGRESSIVE);
            System.out.println("✓ Aggressive config: compression=" + aggressive.compressionType);

            PerformanceOptimizer.OptimizationConfig conservative =
                optimizer.analyze(PerformanceOptimizer.OptimizationProfile.CONSERVATIVE);
            System.out.println("✓ Conservative config: compression=" + conservative.compressionType);

            // Recommend based on metrics
            PerformanceOptimizer.OptimizationConfig recommended = 
                optimizer.recommendConfig(0.7, 0.6);
            assert recommended != null : "Recommendation is null";
            System.out.println("✓ Recommended config: " + recommended);

            // Suggest batch size
            int batch = optimizer.suggestBatchSize(8 * 1024 * 1024 * 1024L, 4 * 1024 * 1024 * 1024L);
            assert batch > 0 : "Invalid batch size";
            System.out.println("✓ Suggested batch size: " + batch);

            // Suggest compression
            String compression = optimizer.suggestCompressionType(2 * 1024 * 1024 * 1024L, 
                                                                  8 * 1024 * 1024 * 1024L);
            System.out.println("✓ Suggested compression: " + compression);

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }

    private static void testSchedulerIntegration() {
        System.out.println("\n[TEST 5] Scheduler Integration");
        System.out.println("─".repeat(50));

        try {
            FeatureExtractor extractor = new FeatureExtractor();
            TrainingDataCollector collector = new TrainingDataCollector();
            NeuralNetworkPredictor predictor = new NeuralNetworkPredictor();

            // Train on sample data first
            List<TrainingSample> trainingData = generateSyntheticData(100);
            predictor.train(trainingData);

            AdaptiveScheduler scheduler = new AdaptiveScheduler(extractor, predictor, collector);

            // Create memory state
            MemoryState state = new MemoryState(
                10,           // current_layer
                1000,         // current_token
                0.7,          // gpu_usage
                0.6,          // ram_usage
                50.0,         // storage_latency
                5,            // cached_layers
                20            // kv_pages
            );

            // Evaluate
            ScheduledDecision decision = scheduler.evaluate(state);
            assert decision != null : "Decision is null";
            assert decision.decision() != null : "No decision returned";
            System.out.println("✓ Evaluation: " + decision.decision());

            // Report result
            scheduler.reportResult(decision, 0.15, 1024 * 1024);
            System.out.println("✓ Result reported");

            // Check training samples collected
            int samples = scheduler.trainingSamples();
            System.out.println("✓ Training samples collected: " + samples);

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }

    private static void testHyperparameterOptimization() {
        System.out.println("\n[TEST 6] Hyperparameter Optimization");
        System.out.println("─".repeat(50));

        try {
            NeuralNetworkPredictor model = new NeuralNetworkPredictor();
            ModelPersistence persistence = new ModelPersistence("test_models");
            MLTrainer trainer = new MLTrainer(model, persistence);

            // Generate synthetic data
            List<TrainingSample> samples = generateSyntheticData(150);

            // Run optimization (with reduced search space for speed)
            model.setTrainingConfig(0.01, 0.001, 10, 32);  // Quick config
            trainer.train(samples);
            System.out.println("✓ Initial training complete");

            // Optimize hyperparameters (this will take a moment)
            System.out.println("  Optimizing hyperparameters (this may take a moment)...");
            trainer.optimizeHyperparameters(samples);
            System.out.println("✓ Hyperparameter optimization complete");

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            failed++;
        }
    }

    private static void testCrossValidation() {
        System.out.println("\n[TEST 7] Cross-Validation");
        System.out.println("─".repeat(50));

        try {
            NeuralNetworkPredictor model = new NeuralNetworkPredictor();
            ModelPersistence persistence = new ModelPersistence("test_models");
            MLTrainer trainer = new MLTrainer(model, persistence);

            // Generate synthetic data
            List<TrainingSample> samples = generateSyntheticData(200);

            // Run 5-fold cross-validation
            trainer.crossValidate(samples, 5);
            System.out.println("✓ 5-fold cross-validation complete");

            Map<String, Double> metrics = trainer.getMetrics();
            double cvAccuracy = metrics.getOrDefault("cv_accuracy", 0.0);
            System.out.println("✓ CV Accuracy: " + String.format("%.2f%%", cvAccuracy));

            passed++;
        } catch (Exception e) {
            System.err.println("✗ FAILED: " + e.getMessage());
            failed++;
        }
    }

    /**
     * Generate synthetic training data for testing.
     */
    private static List<TrainingSample> generateSyntheticData(int count) {
        List<TrainingSample> samples = new ArrayList<>();
        Random rand = new Random(42);

        for (int i = 0; i < count; i++) {
            MemoryState state = new MemoryState(
                rand.nextInt(256),           // layer
                rand.nextInt(100000),         // token
                rand.nextDouble(),            // gpu_usage
                rand.nextDouble(),            // ram_usage
                rand.nextDouble() * 100,      // storage_latency
                rand.nextInt(32),             // cached_layers
                rand.nextInt(1000)            // kv_pages
            );

            SchedulerAction[] actions = SchedulerAction.values();
            SchedulerAction action = actions[rand.nextInt(actions.length)];
            Decision decision = new Decision(action, rand.nextInt(256));

            samples.add(new TrainingSample(state, decision));
        }

        return samples;
    }
}
