[U, S, V] = pagesvd(sampled_image);

S(S<0.05) = 0;

rec_img = pagemtimes(U,S);
rec_img = pagemtimes(rec_img, permute(V,[2 1 3] ));


imshow(rec_img * 255)