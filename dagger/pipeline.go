package main

import (
	"context"
	"log"
	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Connect to Dagger engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Close()

	// Run pipeline
	if err := runPipeline(ctx, client); err != nil {
		log.Fatalf("Pipeline failed: %v", err)
	}

	log.Println("Pipeline completed successfully!")
}

func runPipeline(ctx context.Context, client *dagger.Client) error {

	//  1. Base Python container with repo mounted
	container := client.Container().
		From("python:3.10-slim").
		WithMountedDirectory("/app", client.Host().Directory("..")). // repo root
		WithWorkdir("/app").
		WithEnvVariable("PYTHONPATH", "/app")           // allow "import config"

	//  2. Upgrade pip
	container = container.WithExec([]string{
		"pip", "install", "--upgrade", "pip",
	})

	//  3. Install dependencies
	container = container.WithExec([]string{
		"pip", "install", "-r", "requirements.txt",
	})

	// 4. Install DVC and pull raw data
	container = container.
    	// Install DVC
    	WithExec([]string{
        "pip", "install", "dvc",
    }).
    	// Pull raw_data.csv.dvc, fallback to update if pull fails
    	WithExec([]string{
        "sh", "-c",
        "dvc pull || dvc update data/raw_data.csv.dvc",
    })

	// 5. Run unit tests first
    container = container.WithExec([]string{
        "python", "-m", "unittest", "tests.test_utils",
    })

	//  6. Python scripts to execute in order
	steps := []string{
		"src/01_load.py",
		"src/02_feature_selection.py",
		"src/03_clean_separate.py",
		"src/04_combine_bin_save.py",
		"src/05_train.py",
		"src/06_model_selection.py",
		"src/07_deploy.py",
	}

	//  7. Execute each script
	for _, script := range steps {
		log.Println("Running", script)
		container = container.WithExec([]string{"python", script})
	}

	// 8. Export model artifacts and pipeline artifacts separately
	_, err := container.
		Directory("/app/artifacts").
		Export(ctx, "../artifacts")
	if err != nil {
		return err
	}

	_, err = container.
		Directory("/app/model").
		Export(ctx, "../model")
	if err != nil {
		return err
	}


	// 9. Export validator-compatible model for GitHub Actions
	_, err = container.
		WithExec([]string{
			"bash", "-c",
			"mkdir -p /out/model && cp /app/model/lead_model_lr.pkl /out/model/model.pkl",
		}).
		Directory("/out/model").
		Export(ctx, "../model")   // export to host model folder
	if err != nil {
		return err
	}


	return nil
}

