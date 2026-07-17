use oak_sys::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = Device::list()?;
    if devices.is_empty() {
        println!("No OAK devices found.");
        return Ok(());
    }

    println!("Found {} device(s):", devices.len());
    for dev in devices {
        println!(
            "- id: {}, name: {}, state: {:?}",
            dev.device_id, dev.name, dev.state
        );
    }
    Ok(())
}
