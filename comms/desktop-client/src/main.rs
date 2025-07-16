use std::time::Duration;
use tonic::transport::Channel;

pub mod robot {
    tonic::include_proto!("robot");
}

use robot::robot_service_client::RobotServiceClient;
use robot::PingRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let server_addr = if args.len() > 1 {
        args[1].clone()
    } else {
        "http://localhost:5051".to_string() // Remove this later
    };

    println!("Trying to connect to: {}", server_addr);

    let channel = Channel::from_shared(server_addr)?
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .connect()
        .await?;

    let mut client = RobotServiceClient::new(channel);

    let req = tonic::Request::new(PingRequest {
        message: "Hello from ttrb's mac rust client!".to_string(),
        client_timestamp: chrono::Utc::now().timestamp_millis(),
    });

    println!("Attempting ping...");
    let response = client.ping(req).await?;
    let response_data = response.into_inner();

    let current_time = chrono::Utc::now().timestamp_millis();
    let rtt = current_time - response_data.client_timestamp;

    println!("Got response: {}", response_data.message);
    println!("Server ID: {}", response_data.server_id);
    println!("Round-trip time: {}ms", rtt);

    Ok(())
}
