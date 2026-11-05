use super::*;
fn package(kind:Kind)->Package {
    Package{root:PathBuf::new(),kind,config:json!({"vision_config":{"image_size":4,"patch_size":2,"downsample_ratio":2},"vision_feature_select_strategy":"default"}),
        text:json!({"vocab_size":8,"eos_token_id":7}),vocab:8,image_token:0,seq2seq:false}
}
#[test]
fn native_spans_and_padding_keep_image_identity() {
    let p=package(Kind::V41);
    let image=Picture{rgb:vec![0;4*4*3],width:4,height:4};
    assert_eq!(p.image_types(4,4).unwrap(),vec![0,1,2,3]);
    let (types,start)=image_layout(&p,&image,&[6,0,0,0,0,5,1],&[1.,1.,1.,1.,1.,1.,0.]).unwrap();
    assert_eq!(start,1);assert_eq!(types,vec![-1,0,1,2,3,-1,-1]);
    assert_ne!(p.pad(),p.image_token);
    assert!(image_layout(&p,&image,&[0,0,4,0,0],&[1.;5]).is_err());
    assert!(image_layout(&p,&image,&[0,0,0,0],&[1.,1.,0.,1.]).is_err());
}
#[test]
fn released_v41_rgb_patch_order_and_bfloat_boundary() {
    let picture=Picture{rgb:vec![0,127,255,255,0,127,127,255,0,255,127,0],width:2,height:2};
    let patches=picture.patches(2);
    assert_eq!(&patches[..4],&[-1.,1.,f32::from_bits(0xbb810000),1.]);
    assert_eq!(patches[8],1.);assert_eq!(patches[10],-1.);
}
#[test]
fn partial_batches_do_not_train_repeated_padding_examples() {
    let s=Sample{ids:vec![1,2],mask:vec![1.,1.],decoder:vec![],decoder_mask:vec![],targets:vec![2,3],weights:vec![1.,1.],picture:Picture{rgb:vec![1,2,3],width:1,height:1}};
    let b=Batch::from_samples(&[s],2);
    assert_eq!(b.weights,vec![1.,1.,0.,0.]);assert_eq!(b.rgb,vec![1,2,3,1,2,3]);
}
#[test]
fn vision_options_reject_mismatched_and_invalid_controls() {
    let args=|v:&[&str]|v.iter().map(OsString::from).collect();
    assert!(parse(args(&["--model-path","model","--max-length","0"])).is_err());
    assert!(parse(args(&["--model-path","model","--lr","NaN"])).is_err());
    assert!(parse(args(&["--model-path","model","--native-state","a","--resume","b"])).is_err());
    assert!(parse(args(&["--model-path","model","--cache-implementation","paged"])).is_err());
    let o=parse(args(&["--model-path","model","--image","image with spaces.png","--prompt","look","--device-index","1","--no-do-sample"])).unwrap();
    assert_eq!(o.image.unwrap(),PathBuf::from("image with spaces.png"));assert_eq!(o.device,Some(1));assert!(!o.sample);
}
#[test]
fn shifted_text_target_after_an_image_row_remains_supervised() {
    let directory=env::temp_dir().join(format!("hierarchos-vision-label-{}-{}",std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir(&directory).unwrap();
    image::RgbImage::new(4,4).save(directory.join("image.png")).unwrap();
    let p=package(Kind::Shared);
    let v=json!({"image":"image.png","input_ids":[1,0,0,0,0,7,3,2],"targets":[0,0,0,0,7,3,2,0],"loss_weights":[0.,0.,0.,0.,1.,1.,1.,0.]});
    let sample=Sample::from_record(&v,&directory,&p,None,8).unwrap();assert_eq!(sample.weights[4],1.);assert_eq!(sample.targets[4],7);
    fs::remove_file(directory.join("image.png")).unwrap();fs::remove_dir(&directory).unwrap();
}
