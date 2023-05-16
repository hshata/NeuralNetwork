%HUSSEIN SHATA
%NN 8-3-8 

clear all;
clc;
input= [1 0 0 0 0 0 0 0;0 1 0 0 0 0 0 0;0 0 1 0 0 0 0 0;0 0 0 1 0 0 0 0;0 0 0 0 1 0 0 0;0 0 0 0 0 1 0 0;0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1];
eta= 0.5;
input_to_hidden_weight=rand(8,3);
hidden_output_weight=rand(4,8);
bias=1.0;
iteration_count = 1;
tolerance = 0.01;
a=1;
while (a>tolerance&& iteration_count<300000)
   
    for ii=1:8
        in=input(ii,:);
        hidden_input=in*input_to_hidden_weight;
        hidden_out= 1./(1 + exp(-hidden_input));
        hidden_out;
        aa(ii,:)=hidden_out(1,:);
        if ii==3
         plot_h(iteration_count,:)=hidden_out;    %%For Hidden unit encoding
        end
        plot_w(iteration_count,:)=input_to_hidden_weight(:,2);  %%For the weights from input layer to second hidden unit
        final_hout=[hidden_out bias];
        output_in=final_hout*hidden_output_weight;
        out= 1./(1 + exp(-output_in));
        error_out=out.*(1-out).*(in-out);  
        sq_error= sum((out-in).*(out-in))/2;    %%For sum of squared error
        error(iteration_count,ii)=sq_error;
        hidden_out_w_times_errout=error_out*hidden_output_weight(1:3,:)';      
        error_hidden=hidden_out_w_times_errout.*final_hout(:,1:3).*(1-final_hout(:,1:3));
        delta=eta*in'*error_hidden;
        new_intohidden_weight=input_to_hidden_weight+delta;  %% Updating weights
        delta_hidden_output=eta*final_hout'*error_out;
        new_hiddentooutput_weight=hidden_output_weight+delta_hidden_output;  %% Updating weights
        input_to_hidden_weight=new_intohidden_weight;
        hidden_output_weight=new_hiddentooutput_weight;
        output=out;      
        
         for k=1:8        
             outt(k,ii)=output(k);
         end    
    end
    a=max(error(end,:));
    iteration_count=iteration_count+1;
end
disp(outt)
figure(1)
xlabel('Number of iterations' , 'FontSize', 15);
ylabel('Squared error', 'FontSize', 15);
hold on;
plot(error, 'LineWidth',2,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10);
figure (2)
xlabel('Number of iterations' , 'FontSize', 15);
ylabel('Hidden values', 'FontSize', 15);
hold on;
plot(plot_h, 'LineWidth',2,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10);
figure,(3)
xlabel('Number of iterations' , 'FontSize', 15);
ylabel('Weight values', 'FontSize', 15);
hold on;
plot(plot_w, 'LineWidth',2,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10);
disp(iteration_count)
