function [] = graphs(metrics)
    figure;
    plot(metrics.updates, metrics.loss_train, 'b-', 'LineWidth', 2);
    hold on;
    plot(metrics.updates, metrics.loss_val, 'g-', 'LineWidth', 2);
    hold off;
    title('Training and Validation Loss Over Update Steps', 'FontSize', 18);
    xlabel('Update Step', 'FontSize', 16);
    ylabel('Loss', 'FontSize', 16);
    legend('Training Loss', 'Validation Loss', 'FontSize', 14);
    grid on;
    
    figure;
    plot(metrics.updates, metrics.cost_train, 'b-', 'LineWidth', 2);
    hold on;
    plot(metrics.updates, metrics.cost_val, 'g-', 'LineWidth', 2);
    hold off;
    title('Training and Validation Cost Over Update Steps', 'FontSize', 18);
    xlabel('Update Step', 'FontSize', 16);
    ylabel('Cost', 'FontSize', 16);
    legend('Training Cost', 'Validation Cost', 'FontSize', 14);
    grid on;
    
    figure;
    plot(metrics.updates, metrics.acc_train, 'b-', 'LineWidth', 2);
    hold on;
    plot(metrics.updates, metrics.acc_val, 'g-', 'LineWidth', 2);
    hold off;
    title('Training and Validation Accuracy Over Update Steps', 'FontSize', 18);
    xlabel('Update Step', 'FontSize', 16);
    ylabel('Accuracy (%)', 'FontSize', 16);
    legend('Training Accuracy', 'Validation Accuracy', 'FontSize', 14);
    grid on;
end