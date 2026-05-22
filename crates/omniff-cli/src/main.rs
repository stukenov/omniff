use clap::Parser;

#[derive(Parser)]
#[command(name = "omniff")]
#[command(about = "FFmpeg-like multimodal AI runtime")]
#[command(version)]
struct Cli {
    /// Input file or text
    #[arg(short = 'i', long)]
    input: String,

    /// Prompt / instruction
    #[arg(short, long)]
    prompt: Option<String>,

    /// Output format: text, image, video, audio, document
    #[arg(short = 'f', long = "of")]
    output_format: Option<String>,

    /// Output file path
    #[arg(short, long)]
    output: Option<String>,

    /// Thinking level: off, fast, normal, deep, research
    #[arg(long, default_value = "normal")]
    thinking: String,

    /// Preserve: comma-separated (faces, voice, structure, layout)
    #[arg(long)]
    preserve: Option<String>,

    /// Style/edit strength 0.0-1.0
    #[arg(long)]
    strength: Option<f32>,

    /// Task type override
    #[arg(long)]
    task: Option<String>,

    /// Language hint
    #[arg(long)]
    lang: Option<String>,

    /// Model override (default: auto)
    #[arg(long, default_value = "auto")]
    model: String,

    /// Graph template file
    #[arg(long)]
    graph: Option<String>,

    /// Config file path
    #[arg(long, default_value = "omniff.yaml")]
    config: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    println!("OmniFF Runtime v{}", env!("CARGO_PKG_VERSION"));
    println!("Input:    {}", cli.input);
    if let Some(ref p) = cli.prompt {
        println!("Prompt:   {}", p);
    }
    println!("Thinking: {}", cli.thinking);
    println!("Model:    {}", cli.model);

    // TODO: load config, detect modality, build graph, execute
    eprintln!("Runtime execution not yet implemented — scaffold only.");

    Ok(())
}
