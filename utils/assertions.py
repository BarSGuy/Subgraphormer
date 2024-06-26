def general_assertions(args):
    # Ensure appropriate settings for subgraph sampling and result averaging
    if args.data.sampling.keep_subgraph_prob == 1 and args.data.sampling.average_res_over > 1:
        raise ValueError("If keep_subgraph_prob is 1 (keeping all subgraphs), average_res_over should be 1 to avoid incorrect averaging over multiple results.")

    # Ensure that the architecture is Subgraphormer or Subgraphormer_PE
    if args.model.model_name not in ['Subgraphormer', 'Subgraphormer_PE']:
        raise ValueError("The model architecture must be either 'Subgraphormer' or 'Subgraphormer_PE'.")
