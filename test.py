    for X in train_iter:
                X = resample(X, kernel=kernel)
                        Y = []
                                for i in range(X.shape[1]):
                                                x = X[:, i, :, :, :]
                                                            x = V(FT(x), requires_grad=True).view(-1, 1, 64, 64, 64)
                                                                        x = x.to(devices[0])
                                                                                    y = model(x).view(-1, 64, 64, 64).to(devices[0])
                                                                                                Y.append(y)
                                                                                                        Y_tensor = torch.as_tensor(Y).view(X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4])
                                                                                                                del Y
                                                                                                                        l = 0
                                                                                                                                for i in range(X.shape[1]):
                                                                                                                                                for j in range(X.shape[1]):
                                                                                                                                                                    if i != j:
                                                                                                                                                                                            l += loss(X[:, i, :, :, :], Y[:, j, :, : ,:])
                                                                                                                                                                                                                l += loss(X[:, j, :, :, :], Y[:, i, :, : ,:])
                                                                                                                                                                                                                                    l += loss(Y[:, i, :, :, :], Y[:, j, :, : ,:])
