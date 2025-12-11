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
		WithMountedDirectory("/app", client.Host().Directory("../")). // mount repo root
		WithWorkdir("/app/Module1").                                  // scripts live here
		WithEnvVariable("PYTHONPATH", "/app:/app/Module1")            // allow "import config"

	//  2. Upgrade pip
	container = container.WithExec([]string{
		"pip", "install", "--upgrade", "pip",
	})

	//  3. Install project as editable package
	container = container.WithExec([]string{
		"pip", "install", "-e", "/app",
	})

	//  4. Install dependencies from repo root
	container = container.WithExec([]string{
		"pip", "install", "-r", "/app/requirements.txt",
	})

	// NEW 
	// 4b. Install DVC
	container = container.WithExec([]string{
    "pip", "install", "dvc",
	})

	// 4c. Pull raw_data.csv from DVC
	container = container.WithExec([]string{
    "dvc", "pull", "Module1/artifacts/raw_data.csv.dvc",
	})



	// Run tests
	//log.Println("Running unit tests on test_utils.py...")
	//container = container.WithExec([]string{
	//	"python", "-m", "unittest", "Module1.src.test_utils",
	//})

	//  5. Python scripts to execute in order
	steps := []string{
		"src/01_load.py",
		"src/02_feature_selection.py",
		"src/03_clean_separate.py",
		"src/04_combine_bin_save.py",
		"src/05_train.py",
		"src/06_model_selection.py",
		"src/07_deploy.py",
	}

	//  6. Execute each script
	for _, script := range steps {
		log.Println("Running", script)
		container = container.WithExec([]string{"python", script})
	}

	// 7. Export model artifacts and pipeline artifacts separately
	_, err := container.
		Directory("/app/Module1/models").
		Export(ctx, "models")
	if err != nil {
		return err
	}

	_, err = container.
		Directory("/app/Module1/artifacts").
		Export(ctx, "artifacts")
	if err != nil {
		return err
	}

	return nil
}
