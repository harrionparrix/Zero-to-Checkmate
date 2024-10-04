from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("nets/value.pth", map_location=torch.device('cpu')))
    
    # Create a SummaryWriter
    writer = SummaryWriter()

    # Add model graph to TensorBoard
    dummy_input = torch.rand(1, 5, height, width)  # Adjust the shape based on your input
    writer.add_graph(model, dummy_input)

    # Log model weights and gradients
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, global_step=0)
        writer.add_histogram(name + '_grad', param.grad, global_step=0)

    writer.close()
