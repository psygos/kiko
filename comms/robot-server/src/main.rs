use tonic::{transport::Server, Request, Response, Status};

pub mod robot {
    tonic::include_proto!("robot");
}

use robot::robot_service_server::{RobotService, RobotServiceServer};
use robot::{PingRequest, PingResponse};

#[derive(Debug, Default)]
pub struct KikoService {
    server_id: String,
}

#[tonic::async_trait]
impl RobotService for KikoService {
    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        println!("Got a ping request {:?}", request);

        let input = request.into_inner();

        let reply = PingResponse {
            message: format!("Pong: {}", input.message),
            client_timestamp: input.client_timestamp,
            server_timestamp: chrono::Utc::now().timestamp_millis(),
            server_id: self.server_id.clone(),
        };

        Ok(Response::new(reply))
    }
}

#[tokio::main]

async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kiko_service = KikoService {
        server_id: "jetson-nano".to_string(),
    };

    let addr = "0.0.0.0:5051".parse()?;

    println!("Kiko is listening on {}", addr);

    Server::builder()
        .add_service(RobotServiceServer::new(kiko_service))
        .serve(addr)
        .await?;

    Ok(())
}
