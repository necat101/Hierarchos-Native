// Measured AVX2 Gemma4 decoder: 8 causal tokens, two width-8 heads. The
// forward companion is the measured CPU eager variant of attention_forward.
// This header uses the parent shader's bindings and AttentionBackwardPush ABI.
bool gemma4_cpu_text_profile(){
    return pc.batch_size==1u && pc.query_seq_len==8u && pc.key_value_seq_len==8u
        && pc.num_heads==2u && pc.num_key_value_heads==2u
        && pc.query_hidden_size==16u && pc.value_hidden_size==16u
        && pc.causal==1u && pc.window_size==0u && pc.unfused_qk==1u
        && pc.dropout_p==0.0 && pc.has_sink==0u && pc.value_scale==1.0
        && pc.softcap==0.0 && pc.alibi_mode==0u && pc.relative_bias_mode==0u
        && pc.has_sparse_mask==0u && pc.compressed_key_count==0u && pc.paged_kv==0u
        && pc.query_position_offset==0u && pc.key_block_index==0u && pc.key_block_count==1u;
}
float gemma4_text_madd(float a,float b,float c,bool unfused){
    if(unfused){precise float product=a*b;precise float result=c+product;return result;}
    return fp32_fma_exact(a,b,c);
}
float gemma4_text_p[64];float gemma4_text_ds[64];
void gemma4_cpu_text_backward(){
    uint head=gl_GlobalInvocationID.x,batch=gl_GlobalInvocationID.y;
    if(head>=2u || batch>=1u)return;
    for(uint row=0u;row<8u;++row){
        if(query_mask[batch*8u+row]<=0.0){
            for(uint col=0u;col<8u;++col){gemma4_text_p[row*8u+col]=0.0;gemma4_text_ds[row*8u+col]=0.0;}
            continue;
        }
        float maximum=-3.402823466e+38;
        for(uint col=0u;col<8u;++col){
            uint z=row*8u+col;
            if(col>row || key_mask[batch*8u+col]<=0.0){gemma4_text_p[z]=-3.402823466e+38;gemma4_text_ds[z]=0.0;continue;}
            float a=0.0,b=0.0,da=0.0,db=0.0;
            for(uint d=0u;d<8u;++d){
                uint qi=(batch*8u+row)*16u+head*8u+d,ki=(batch*8u+col)*16u+head*8u+d;
                if((d&1u)==0u){a=gemma4_text_madd(q_values[qi],k_values[ki],a,row>=6u);da=gemma4_text_madd(grad_output[qi],v_values[ki],da,row>=6u);}
                else {b=gemma4_text_madd(q_values[qi],k_values[ki],b,row>=6u);db=gemma4_text_madd(grad_output[qi],v_values[ki],db,row>=6u);}
            }
            gemma4_text_p[z]=(a+b)*pc.scale;gemma4_text_ds[z]=da+db;maximum=max(maximum,gemma4_text_p[z]);
        }
        precise float lanes[8];for(uint col=0u;col<8u;++col){
            uint z=row*8u+col;gemma4_text_p[z]=(col<=row && key_mask[batch*8u+col]>0.0)
                ? sleef_expf_u10(gemma4_text_p[z]-maximum):0.0;lanes[col]=gemma4_text_p[z];
        }
        precise float total=0.0;for(uint j=0u;j<8u;++j)total+=lanes[j];
        float inverse=fp32_div(1.0,total);
        for(uint col=0u;col<8u;++col){uint z=row*8u+col;gemma4_text_p[z]*=inverse;lanes[col]=gemma4_text_p[z]*gemma4_text_ds[z];}
        precise float dot=0.0;for(uint j=0u;j<8u;++j)dot+=lanes[j];
        for(uint col=0u;col<8u;++col){uint z=row*8u+col;precise float delta=gemma4_text_ds[z]-dot;
            precise float value=gemma4_text_p[z]*delta;gemma4_text_ds[z]=value*pc.scale;}
    }
    for(uint row=0u;row<8u;++row)for(uint d=0u;d<8u;++d){
        float qa=0.0,qb=0.0,ka=0.0,kb=0.0,va=0.0,vb=0.0;
        for(uint j=0u;j<8u;++j){
            uint z=(batch*8u+j)*16u+head*8u+d;
            if((j&1u)==0u){
                qa=gemma4_text_madd(gemma4_text_ds[row*8u+j],k_values[z],qa,row>=6u);
                ka=gemma4_text_madd(q_values[z],gemma4_text_ds[j*8u+row],ka,d>=6u);
                va=gemma4_text_madd(gemma4_text_p[j*8u+row],grad_output[z],va,row>=6u);
            }else{
                qb=gemma4_text_madd(gemma4_text_ds[row*8u+j],k_values[z],qb,row>=6u);
                kb=gemma4_text_madd(q_values[z],gemma4_text_ds[j*8u+row],kb,d>=6u);
                vb=gemma4_text_madd(gemma4_text_p[j*8u+row],grad_output[z],vb,row>=6u);
            }
        }
        uint z=(batch*8u+row)*16u+head*8u+d;grad_q[z]=qa+qb;grad_k[z]=ka+kb;grad_v[z]=va+vb;
    }
}
