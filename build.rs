fn main() {
    ::capnpc::CompilerCommand::new().file("src/hello_world.capnp").run().unwrap();
}